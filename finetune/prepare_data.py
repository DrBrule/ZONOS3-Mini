"""Prepare one speaker's recordings for fine-tuning: codec codes, speaker embeddings, normalized text, conditions.

    uv run --package zonos3-sgl python finetune/prepare_data.py --audio-dir recordings/ --out data/myvoice \
        [--metadata metadata.csv | sidecar <clip>.txt files] --language en_us

Transcripts come from ``--metadata`` (``file|text`` or ``file<TAB>text`` lines, LJSpeech style, or JSONL with
``audio`` and ``text``) or, without it, a ``.txt`` / ``.lab`` file next to each audio file. Each clip is mixed to mono,
resampled to 44.1 kHz, trimmed of leading / trailing silence (keeping ``--keep-silence`` seconds), then

- encoded with the matched dual codec (zonos3.tokenizer.dual_reference, CUDA) to int16 codes [frames, 10] at 15 Hz;
- embedded with the checkpoint's speaker encoder (MECT ``ownmect_s2``, 192-D);
- its text normalized like the server's requests (zyphra_textnorm), so training sees what inference sends;
- measured for the condition rows: loudness (LUFS, pyloudnorm), speaking rate (UTF-8 bytes of the normalized text per
  second) and an estimated bandwidth ``fmax`` (Hz, heuristic; train.py leaves fmax unknown by default).

Output: ``manifest.jsonl``, ``codes/<id>.npy``, ``speaker_embeddings.npy``, ``speaker_mean.npy`` (unit-normalized
mean scaled to the clips' mean norm) and ``dataset.json``; ``--val`` clips go to the ``val`` split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ft_runtime import RELEASE_LM, embed, load_audio, speaker_encoder  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac"}
RATE = 44100
HOP = 2940


def read_transcripts(args) -> list[tuple[Path, str]]:
    audio_dir = Path(args.audio_dir)
    pairs = []
    if args.metadata:
        meta = Path(args.metadata)
        if meta.suffix == ".jsonl":
            for line in meta.read_text().splitlines():
                if line.strip():
                    m = json.loads(line)
                    pairs.append((m["audio"], m["text"]))
        else:
            delim = "\t" if "\t" in meta.read_text().splitlines()[0] else "|"
            with meta.open(newline="") as f:
                for row in csv.reader(f, delimiter=delim, quoting=csv.QUOTE_NONE):
                    if len(row) >= 2 and row[0].strip():
                        pairs.append((row[0].strip(), row[-1].strip()))
        out = []
        for name, text in pairs:
            path = audio_dir / name
            if not path.suffix:
                path = next((path.with_suffix(e) for e in AUDIO_EXT if path.with_suffix(e).exists()), path)
            out.append((path, text))
        return out
    for path in sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXT):
        side = next((path.with_suffix(e) for e in (".txt", ".lab") if path.with_suffix(e).exists()), None)
        if side is not None:
            out_text = side.read_text().strip()
            if out_text:
                pairs.append((path, out_text))
    return pairs


def trim(wav: torch.Tensor, keep: float, top_db: float) -> torch.Tensor:
    """Cut leading / trailing audio more than ``top_db`` below the clip's loudest 20 ms frame, keeping ``keep`` s."""
    frame = int(0.02 * RATE)
    n = wav.shape[-1] // frame
    if n < 3:
        return wav
    energy = wav[0, : n * frame].reshape(n, frame).square().mean(-1).clamp(min=1e-12).log10() * 10
    loud = (energy > energy.max() - top_db).nonzero().flatten()
    start = max(0, int(loud[0]) * frame - int(keep * RATE))
    end = min(wav.shape[-1], (int(loud[-1]) + 1) * frame + int(keep * RATE))
    return wav[:, start:end]


def loudness_lufs(wav: np.ndarray) -> float | None:
    try:
        import pyloudnorm

        value = pyloudnorm.Meter(RATE).integrated_loudness(wav.astype(np.float64))
        return float(value) if math.isfinite(value) else None
    except Exception:  # noqa: BLE001
        return None


def estimate_fmax(wav: np.ndarray, floor_db: float = 50.0) -> float:
    """Highest frequency whose long-term spectrum is within ``floor_db`` of the speech band's peak (a bandwidth
    estimate: ~8 kHz for 16 kHz sources upsampled, ~20 kHz for full-band recordings)."""
    n_fft = 4096
    if len(wav) < n_fft:
        return RATE / 2
    frames = np.lib.stride_tricks.sliding_window_view(wav, n_fft)[:: n_fft // 2]
    spec = (np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=-1)) ** 2).mean(0)
    db = 10 * np.log10(spec + 1e-20)
    freqs = np.fft.rfftfreq(n_fft, 1 / RATE)
    db = np.convolve(db, np.ones(9) / 9, mode="same")
    peak = db[(freqs > 200) & (freqs < 4000)].max()
    above = np.nonzero(db > peak - floor_db)[0]
    return float(freqs[above[-1]]) if len(above) else RATE / 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--metadata", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--language", default="en_us", help="server language code used for text normalization")
    p.add_argument("--model", default=RELEASE_LM, help="base LM (speaker encoder and context length)")
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--keep-silence", type=float, default=0.15, help="silence kept around trimmed speech (s)")
    p.add_argument("--top-db", type=float, default=45.0)
    p.add_argument("--no-trim", action="store_true")
    p.add_argument("--no-normalize-text", action="store_true", help="keep transcripts as written")
    p.add_argument("--val", type=int, default=10, help="held-out clips")
    p.add_argument("--codec-device", default="cuda:0")
    p.add_argument("--speaker-device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    from zonos3.reference.lm import Zonos3LMConfig
    from zonos3.tokenizer.dual_reference import encode_dual_reference
    from zonos3.tokenizer.textnorm import load_normalizer, normalize_request_text
    from zonos3.utils.release import resolve_dir_or_local

    root = Path(resolve_dir_or_local(a.model))
    config = Zonos3LMConfig.from_params(json.loads((root / "params.json").read_text()))

    class _Cfg:  # what speaker_encoder needs from an LM
        def __init__(self, c):
            self.config = c

    encoder = speaker_encoder(_Cfg(config), a.speaker_device)
    normalizer = None if a.no_normalize_text else load_normalizer()

    out = Path(a.out)
    (out / "codes").mkdir(parents=True, exist_ok=True)
    pairs = read_transcripts(a)
    print(f"{len(pairs)} clips with transcripts")
    rows, embeddings, skipped = [], [], {}
    used_ids = set()
    for path, raw_text in pairs:
        try:
            wav, sr = load_audio(path)
        except Exception as exc:  # noqa: BLE001
            skipped[str(path)] = f"unreadable: {exc}"
            continue
        wav = wav.float().mean(0, keepdim=True)
        if sr != RATE:
            import torchaudio.functional as AF

            wav = AF.resample(wav, sr, RATE)
        if not a.no_trim:
            wav = trim(wav, a.keep_silence, a.top_db)
        peak = float(wav.abs().max())
        if peak > 1.0:
            wav = wav / peak
        seconds = wav.shape[-1] / RATE
        if not a.min_seconds <= seconds <= a.max_seconds:
            skipped[str(path)] = f"{seconds:.1f} s outside [{a.min_seconds}, {a.max_seconds}]"
            continue
        text = raw_text if normalizer is None else normalize_request_text(raw_text, a.language, normalizer)
        codes = encode_dual_reference(wav, RATE, max_seconds=math.ceil(seconds) + 1, device=a.codec_device)
        codes = np.asarray(codes.cpu() if torch.is_tensor(codes) else codes).astype(np.int16)
        if codes.ndim != 2 or codes.shape[1] != config.n_codebooks:
            skipped[str(path)] = f"unexpected codes {codes.shape}"
            continue
        clip_id = re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem)
        while clip_id in used_ids:
            clip_id += "_"
        used_ids.add(clip_id)
        np.save(out / "codes" / f"{clip_id}.npy", codes)
        mono = wav[0].numpy()
        embeddings.append(embed(encoder, wav, RATE, config.speaker_embedding_dim))
        rows.append({
            "id": clip_id, "audio": str(path), "raw_text": raw_text, "text": text, "seconds": round(seconds, 3),
            "frames": int(codes.shape[0]), "embedding_index": len(embeddings) - 1,
            "conditions": {"loudness": loudness_lufs(mono),
                           "speaking_rate": round(len(text.encode("utf-8")) / seconds, 3),
                           "fmax": estimate_fmax(mono)},
        })
        print(f"{clip_id}: {seconds:.1f} s, {codes.shape[0]} frames")

    if not rows:
        raise SystemExit(f"no usable clips (skipped: {json.dumps(skipped, indent=1)})")
    rng = random.Random(a.seed)
    val = set(rng.sample(range(len(rows)), min(a.val, max(0, len(rows) - 1))))
    for i, r in enumerate(rows):
        r["split"] = "val" if i in val else "train"
    embs = np.stack(embeddings).astype(np.float32)
    train_idx = [i for i in range(len(rows)) if i not in val]
    train_embs = embs[train_idx]
    unit = train_embs / np.linalg.norm(train_embs, axis=1, keepdims=True)
    mean = unit.mean(0)
    mean = mean / np.linalg.norm(mean) * float(np.linalg.norm(train_embs, axis=1).mean())
    np.save(out / "speaker_embeddings.npy", embs)
    np.save(out / "speaker_mean.npy", mean.astype(np.float32))
    with (out / "manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(r["seconds"] for r in rows) / 3600
    sims = unit @ (mean / np.linalg.norm(mean))
    summary = {"language": a.language, "model": a.model, "clips": len(rows), "val": len(val), "hours": round(hours, 3),
               "skipped": len(skipped), "speaker_sim_to_mean": {"mean": float(sims.mean()), "min": float(sims.min())}}
    for name in ("loudness", "speaking_rate", "fmax"):
        vals = [r["conditions"][name] for r in rows if r["conditions"][name] is not None]
        if vals:
            summary[name] = {"median": float(np.median(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    (out / "dataset.json").write_text(json.dumps({**summary, "skipped_files": skipped}, indent=2))
    print(json.dumps(summary, indent=2))
    low = [rows[i]["id"] for i, s in zip(train_idx, sims) if s < 0.5]
    if low:
        print(f"warning: {len(low)} clips sound unlike the rest (speaker cosine < 0.5), check them: {low[:10]}")


if __name__ == "__main__":
    main()
