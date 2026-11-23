"""Generate samples from a fine-tuned checkpoint and score speaker similarity.

    uv run --package zonos3-sgl python finetune/sample.py --checkpoint runs/myvoice/best --data data/myvoice \
        --out runs/myvoice/samples --compare-base --text "Any sentence you like."

Synthesis is the plain-PyTorch reference (zonos3.reference.synth.synthesize: the server's prompt plan, sampler and
codec decode) on the merged weights. For each text it writes ``<mode>_<i>.wav`` for the speaker-embedding path (the
checkpoint's speaker vector) and the raw-reference path (a held-out clip's codes as the prefix), and with
``--compare-base`` the same requests on the base LM (the base gets the dataset mean embedding). ``results.json``
holds the cosine similarity of each take's MECT speaker embedding to the dataset's mean embedding (higher = closer to
the target speaker; compare fine-tuned against base, and against real clips' spread printed as ``data_sim``).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ft_data import load_dataset  # noqa: E402
from ft_runtime import (  # noqa: E402
    cosine,
    embed,
    load_base,
    load_checkpoint,
    merged_lm,
    npy_bytes,
    speaker_encoder,
    speaker_vector,
)

SAMPLE_RATE = 44100


def _requests(text, language, seed, speaker, reference_codes):
    base = {"text": text, "language": language, "seed": seed}
    out = {}
    if speaker is not None:
        out["embedding"] = {**base, "speaker_mode": "embedding",
                            "speaker_embedding_base64": base64.b64encode(npy_bytes(speaker)).decode()}
    if reference_codes is not None:
        out["reference"] = {**base, "speaker_mode": "reference",
                            "speaker_reference_codes": reference_codes.tolist()}
    return out


@torch.inference_mode()
def run(lm, tag, texts, language, speaker, reference_codes, out: Path, encoder, mean, seed=1):
    from zonos3.reference.synth import ReferenceAssets, synthesize
    from zonos3.tts.audio import write_pcm16_wav

    assets = ReferenceAssets.load()
    modes = ["embedding"] + (["reference"] if lm.config.speaker_reference_prefix_enabled else [])
    results = []
    for i, text in enumerate(texts):
        for mode, req in _requests(text, language, seed + i, speaker,
                                   reference_codes if "reference" in modes else None).items():
            res = synthesize(lm, req, assets=assets)
            path = out / f"{tag}_{mode}_{i}.wav"
            write_pcm16_wav(path, res.pcm, SAMPLE_RATE)
            sim = None
            if encoder is not None and len(res.pcm) > SAMPLE_RATE // 2:
                emb = embed(encoder, torch.from_numpy(res.pcm)[None], SAMPLE_RATE, lm.config.speaker_embedding_dim)
                sim = cosine(emb, mean)
            results.append({"model": tag, "mode": mode, "text": text, "wav": path.name,
                            "seconds": len(res.pcm) / SAMPLE_RATE, "sim": sim})
            print(json.dumps(results[-1]), flush=True)
    return results


def summarize(results):
    out = {}
    for r in results:
        if r["sim"] is not None:
            out.setdefault(f"{r['model']}_{r['mode']}", []).append(r["sim"])
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}


def generate_samples(out, texts, language, *, base, ft=None, ckpt=None, speaker, device, reference, data,
                     compare_base=False, sim=True):
    """Samples of the fine-tuned model (an in-memory ``FTModel`` or a checkpoint dict) into ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if ft is not None:
        lm = load_base(base, device, torch.float32)
        lm.load_state_dict(ft.merged_state_dict(), strict=True)
        lm.cast(dtype)
    else:
        lm = merged_lm(ckpt, device, dtype, base=base)
    clips, mean = load_dataset(data)
    encoder = speaker_encoder(lm, "cpu") if sim else None
    ref_codes = reference.codes()[: int(10 * 15)] if reference is not None else None
    results = run(lm, "finetuned", texts, language, speaker, ref_codes, out, encoder, mean)
    del lm
    if compare_base:
        base_lm = load_base(base, device, dtype)
        results += run(base_lm, "base", texts, language, mean, ref_codes, out, encoder, mean)
    summary = summarize(results)
    if encoder is not None:
        train_embs = [c.embedding for c in clips if c.split == "train"]
        summary["data_sim"] = round(float(np.mean([cosine(e, mean) for e in train_embs])), 4)
    (out / "results.json").write_text(json.dumps({"summary": summary, "takes": results}, indent=2))
    print(json.dumps(summary))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="training checkpoint folder (adapter.pt)")
    p.add_argument("--data", required=True, help="prepared dataset (mean embedding, reference clip)")
    p.add_argument("--out", required=True)
    p.add_argument("--text", action="append", default=[])
    p.add_argument("--language", default=None)
    p.add_argument("--base", default=None, help="base LM (default: the one the checkpoint was trained on)")
    p.add_argument("--compare-base", action="store_true")
    p.add_argument("--no-sim", action="store_true", help="skip the speaker-similarity scoring")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    ckpt = load_checkpoint(a.checkpoint)
    clips, _ = load_dataset(a.data)
    val = [c for c in clips if c.split == "val"] or clips
    texts = a.text or [c.text for c in val[:3]]
    generate_samples(a.out, texts, a.language or ckpt["config"].get("language", "en_us"),
                     base=a.base or ckpt["config"]["base"], ckpt=ckpt, speaker=speaker_vector(ckpt), device=a.device,
                     reference=val[-1], data=a.data, compare_base=a.compare_base, sim=not a.no_sim)


if __name__ == "__main__":
    main()
