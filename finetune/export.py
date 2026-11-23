"""Export a fine-tuning checkpoint for serving.

    uv run --package zonos3-sgl python finetune/export.py --checkpoint runs/myvoice/best --out exports/myvoice --name myvoice

Writes into ``--out``:

- ``model/``: the base checkpoint folder (params.json, tokenizer, ...) with the adapters merged into ``model.pth``
  (same tensor names and dtypes). Serve it with ``SERVE_MODEL=<out>/model servers/sgl/scripts/serve.sh`` (or
  ``--model-path``), run it with ``python -m zonos3.reference --model <out>/model``, or convert it for MLX / ggml / web
  with ``tools/convert_gguf.py lm``.
- ``voices/<name>.npy``: the trained speaker vector, a default voice for ``--default-voices-dir`` / the studio, or a
  ``speaker_embedding_base64`` upload.
- ``accent/<name>.pt`` (LoRA runs only): the LoRA as a ``zonos-accent-lora/1`` accent adapter. Serve the *base* LM with
  ``--accent-adapters <out>/accent`` and send ``"accent": "<name>"`` to switch the fine-tune on per request (strength 1 =
  the merged model).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ft_runtime import ft_from_checkpoint, load_base, load_checkpoint, speaker_vector  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="training checkpoint folder (adapter.pt)")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="finetuned", help="voice / accent name")
    p.add_argument("--base", default=None, help="base LM (default: the one the checkpoint was trained on)")
    p.add_argument("--no-merge", action="store_true", help="skip the merged model folder")
    a = p.parse_args(argv)

    from zonos3.utils.release import resolve_dir_or_local

    ckpt = load_checkpoint(a.checkpoint)
    base = a.base or ckpt["config"]["base"]
    out = Path(a.out)
    (out / "voices").mkdir(parents=True, exist_ok=True)
    np.save(out / "voices" / f"{a.name}.npy", speaker_vector(ckpt))

    lm = load_base(base, "cpu", torch.float32)
    ft = ft_from_checkpoint(lm, ckpt, gdn_backend="torch")
    if ckpt["config"]["mode"] == "lora":
        (out / "accent").mkdir(exist_ok=True)
        torch.save(ft.accent_adapter(), out / "accent" / f"{a.name}.pt")

    if not a.no_merge:
        root = Path(resolve_dir_or_local(base))
        model_dir = out / "model"
        shutil.copytree(root, model_dir, ignore=shutil.ignore_patterns("model.pth"), dirs_exist_ok=True)
        original = torch.load(root / "model.pth", map_location="cpu", mmap=True, weights_only=True)
        merged = ft.merged_state_dict()
        merged = {k: v.to(original[k].dtype).contiguous() for k, v in merged.items()}
        torch.save(merged, model_dir / "model.pth")
        (model_dir / "finetune.json").write_text(json.dumps(
            {"base": base, "checkpoint": str(a.checkpoint), "step": ckpt.get("step"), **ckpt["config"]}, indent=2))
    print(f"exported to {out}")


if __name__ == "__main__":
    main()
