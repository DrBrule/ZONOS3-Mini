"""Dataset and training-sequence layout for speaker fine-tuning.

A prepared dataset (``prepare_data.py``) is a folder with ``manifest.jsonl`` (one clip per line: id, normalized text,
seconds, frames, split, raw condition values), ``codes/<id>.npy`` (int16 [frames, 10] dual-codec codes) and
``speaker_embeddings.npy`` ([clips, 192] MECT embeddings, manifest order) + ``speaker_mean.npy``.

One training sequence is the serving prompt followed by the target audio, exactly as the runtime lays it out
(zonos3.tts.plan.prompt_rows + silence prefill):

    embedding mode:  [speaker row][condition rows][BOS text EOS][silence rows][delayed audio ... EOA ... tail]
    reference mode:  [delayed reference prefix  ][condition rows][BOS text EOS][silence rows][delayed audio ...]

The audio is the clip's codes with ``silence_frames`` codec silence frames in front and one all-EOA frame behind,
sheared (codebook j delayed by j rows), text column = the text pad. Logits at row i predict row i + 1; only rows after
the silence prefill are supervised, and audio-pad targets are ignored.
"""

from __future__ import annotations

import bisect
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from zonos3.tts.prompt import TTSPromptBuilder, shear

IGNORE = -100


@dataclass
class Clip:
    id: str
    text: str
    seconds: float
    frames: int
    split: str
    conditions: dict
    codes_path: Path
    embedding: np.ndarray

    def codes(self) -> torch.Tensor:
        return torch.from_numpy(np.load(self.codes_path).astype(np.int64))


def load_dataset(root) -> tuple[list[Clip], np.ndarray]:
    root = Path(root)
    embeddings = np.load(root / "speaker_embeddings.npy")
    clips = []
    for i, line in enumerate((root / "manifest.jsonl").read_text().splitlines()):
        if not line.strip():
            continue
        m = json.loads(line)
        clips.append(
            Clip(
                m["id"], m["text"], float(m["seconds"]), int(m["frames"]), m.get("split", "train"),
                m.get("conditions", {}), root / "codes" / f"{m['id']}.npy", embeddings[m["embedding_index"]],
            )
        )
    return clips, np.load(root / "speaker_mean.npy")


def condition_ids(config, values: dict, use: set[str], dropout: float, rng: random.Random) -> list[int] | None:
    """One bucket id per condition table: a raw value through the checkpoint's edges (bisect_right, as the runtime)
    when the condition is in ``use`` and survives ``dropout``, else the table's unknown id."""
    names = config.condition_names
    if not names:
        return None
    out = []
    for i, name in enumerate(names):
        unknown = config.condition_bucket_counts[i]
        value, edges = values.get(name), config.condition_edges.get(name)
        if name in use and value is not None and edges and rng.random() >= dropout:
            out.append(int(bisect.bisect_right(list(edges), float(value))))
        else:
            out.append(unknown)
    return out


@dataclass
class Example:
    ids: torch.Tensor  # [rows, n_codebooks + 1]
    targets: torch.Tensor  # [rows, n_codebooks], IGNORE where unsupervised (row i = target of logits row i)
    speaker_row: int  # -1 = no speaker row
    condition_ids: list[int] | None
    condition_start: int


def build_example(
    config,
    builder: TTSPromptBuilder,
    text: str,
    codes: torch.Tensor,
    *,
    silence_frame: torch.Tensor | None,
    silence_frames: int,
    with_speaker_row: bool,
    reference_codes: torch.Tensor | None = None,
    conditions: list[int] | None = None,
) -> Example:
    n, pad = config.n_codebooks, config.audio_pad_id
    text_rows = builder.build(text).long()
    parts = []
    if with_speaker_row:
        parts.append(builder.speaker_slot(dtype=torch.long))
    if reference_codes is not None:
        parts.append(builder.speaker_reference_prefix(reference_codes.to(torch.int32)).long())
    condition_start = sum(len(p) for p in parts) if conditions is not None else -1
    if conditions is not None:
        parts.append(builder.condition_rows(len(conditions), dtype=torch.long))
    parts.append(text_rows)
    prompt_len = sum(len(p) for p in parts)

    frames = [codes.long()]
    p = silence_frames if silence_frame is not None else 0
    if p:
        frames.insert(0, silence_frame.long().reshape(1, n).repeat(p, 1))
    frames.append(torch.full((1, n), config.eoa_id, dtype=torch.long))
    frames.append(torch.full((n - 1, n), pad, dtype=torch.long))  # flush: every codebook's EOA lands in a row
    audio = shear(torch.cat(frames), pad)
    audio = torch.cat((audio, torch.full((len(audio), 1), config.text_vocab, dtype=torch.long)), 1)
    ids = torch.cat(parts + [audio])

    targets = torch.full((len(ids), n), IGNORE, dtype=torch.long)
    first = prompt_len + p  # first supervised row
    nxt = ids[first:, :n]
    targets[first - 1 : len(ids) - 1] = torch.where(nxt == pad, torch.full_like(nxt, IGNORE), nxt)
    return Example(ids, targets, 0 if with_speaker_row else -1, conditions, condition_start)


def random_reference(clips: list[Clip], exclude: str, rng: random.Random, seconds: tuple[float, float], frame_rate: float,
                     max_frames: int) -> torch.Tensor | None:
    """A random 3-10 s (``seconds``) window of another training clip of the speaker: the raw reference prefix."""
    others = [c for c in clips if c.id != exclude and c.split == "train"]
    if not others:
        return None
    clip = rng.choice(others)
    codes = clip.codes()
    want = int(rng.uniform(*seconds) * frame_rate)
    want = max(1, min(want, len(codes), max_frames))
    start = rng.randint(0, len(codes) - want)
    return codes[start : start + want]
