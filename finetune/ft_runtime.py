"""Shared loading helpers: the base LM, training checkpoints, the merged fine-tuned LM and the speaker encoder."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch

from ft_model import FTModel
from zonos3.reference.lm import Zonos3LM

RELEASE_LM = "hf://Zyphra/ZONOS3-Mini/lm/zonos3_mini"


def load_base(model: str, device, dtype: torch.dtype) -> Zonos3LM:
    lm = Zonos3LM.load(model, device=device, dtype=dtype)
    return lm


def load_checkpoint(path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "adapter.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def ft_from_checkpoint(lm: Zonos3LM, ckpt: dict, gdn_backend: str = "auto") -> FTModel:
    a = ckpt["config"]
    ft = FTModel(lm, mode=a["mode"], targets=a["targets"], rank=a["rank"], alpha=a["alpha"], gdn_backend=gdn_backend)
    ft.load_trainable_state(ckpt["trainable"])
    return ft


def merged_lm(ckpt: dict, device, dtype: torch.dtype, base: str | None = None) -> Zonos3LM:
    """The base LM of ``ckpt`` with its adapters folded into the weights (a plain ``Zonos3LM``, ready for
    ``zonos3.reference.synth.synthesize``)."""
    lm = load_base(base or ckpt["config"]["base"], device, torch.float32)
    ft = ft_from_checkpoint(lm, ckpt, gdn_backend="torch")
    state = ft.merged_state_dict()
    lm.load_state_dict(state, strict=True)
    return lm.cast(dtype)


def speaker_vector(ckpt: dict) -> np.ndarray:
    return np.asarray(ckpt["speaker"], dtype=np.float32).reshape(-1)


def npy_bytes(vec: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(vec, dtype=np.float32))
    return buf.getvalue()


def speaker_encoder(lm: Zonos3LM, device: str = "cpu"):
    from zonos3.models.speaker_cloning import create_speaker_embedder

    c = lm.config
    return create_speaker_embedder(c.speaker_embedding_codec, c.speaker_embedding_dim, device=device)


@torch.inference_mode()
def embed(encoder, wav: torch.Tensor, sample_rate: int, dim: int) -> np.ndarray:
    out = encoder(wav, sample_rate)
    for cand in out if isinstance(out, tuple) else (out,):
        cand = cand.squeeze(0).float().cpu()
        if cand.numel() == dim:
            return cand.numpy()
    raise ValueError(f"speaker encoder returned no {dim}-dim embedding")


def load_audio(path) -> tuple[torch.Tensor, int]:
    """[channels, samples] float32 and its rate (soundfile; other formats through the runtime's ffmpeg transcode)."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), int(sr)
    except Exception:  # noqa: BLE001
        from zonos3.tts.audio import _decode_audio_bytes

        wav, sr = _decode_audio_bytes(Path(path).read_bytes())
        wav = torch.as_tensor(wav, dtype=torch.float32)
        return (wav if wav.ndim == 2 else wav[None]), int(sr)


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
