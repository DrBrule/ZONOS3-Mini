# Speaker fine-tuning for ZONOS3 Mini

Adapt the release LM (`hf://Zyphra/ZONOS3-Mini/lm/zonos3_mini`, 886M params, ~100M active) to reproduce one speaker.
The codec, speaker encoder and decoder stay frozen; only the LM is trained (LoRA by default).

Run everything from the repo root inside the CUDA workspace (`uv sync --locked --all-packages`), e.g.
`uv run --package zonos3-sgl python finetune/<script>.py ...`. A GPU is needed for data prep (the dual codec encoder
is CUDA-only) and recommended for training. The GDN layers use FLA's `chunk_gated_delta_rule` (installed with the sgl
server) when it passes a parity check at startup, else a plain-PyTorch chunked kernel.

## 1. Data

Clean recordings of the speaker with exact transcripts. 20 minutes works; 1 to 3 hours is better. Clips of 2 to 20 s,
one speaker, no music, consistent mic. Transcripts come from a `file|text` (or tab-separated) list, a JSONL with
`audio`/`text`, or a `.txt` file next to each audio file.

```bash
uv run --package zonos3-sgl python finetune/prepare_data.py \
  --audio-dir recordings/ --metadata recordings/metadata.csv --language en_us --out data/myvoice
```

It trims silence, encodes 44.1 kHz audio to the LM's 10x4096 codes at 15 Hz, computes MECT speaker embeddings
(per clip and a mean), normalizes text exactly as the server does, measures loudness / speaking rate / bandwidth for
the condition rows, and holds out `--val 10` clips. Check `dataset.json`: `speaker_sim_to_mean` should be high
(above ~0.7) and it lists clips that sound unlike the rest.

## 2. Train

```bash
uv run --package zonos3-sgl python finetune/train.py --data data/myvoice --out runs/myvoice
```

Defaults: LoRA rank 32 on attention, GDN, dense FFN and the MoE latent projections (experts and router frozen),
lr 1e-4 with warmup and cosine decay, 8 clips per step, 6 epochs. Each sequence is the server's prompt plus the
target audio in the training layout (speaker row or raw reference prefix, condition rows, text, silence prefill,
delayed codes, EOA), with next-row cross entropy over all 10 codebooks. Useful flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--mode lora\|full` | `lora` | `full` trains all non-expert weights (fp32 master weights, bf16 autocast), lr 1e-5; add `--train-experts` for the experts too |
| `--rank`, `--alpha`, `--targets` | 32, 32, see `ft_model.DEFAULT_TARGETS` | add `multi_output`, `router.router_linear` for more capacity |
| `--speaker-mode` | `mixed` | `embedding` (speaker row), `reference` (prefix from another clip of the speaker) or both at random, so both inference paths learn the voice |
| `--speaker-vector` | `learned` | trainable speaker embedding, initialized at the dataset mean and exported as `speaker.npy`; `mean` keeps it frozen, `clip` uses each clip's own |
| `--conditions` | `loudness,speaking_rate` | measured condition rows (30% dropped to unknown); fmax stays unknown since its estimate is heuristic |
| `--steps` / `--epochs` | 6 epochs | watch `val_loss_*` in `log.jsonl`; `runs/<name>/best` keeps the lowest embedding-mode val loss |
| `--grad-checkpoint` | off | for long clips or small GPUs |
| `--sample-every N`, `--sample-text` | end only | writes WAVs plus speaker similarity next to the checkpoint |

Overfitting shows up as val loss rising while train loss keeps falling; with under an hour of audio try
`--epochs 3` or `--rank 16`.

## 3. Listen and score

```bash
uv run --package zonos3-sgl python finetune/sample.py --checkpoint runs/myvoice/best --data data/myvoice \
  --out runs/myvoice/samples --compare-base --text "A sentence the speaker never recorded."
```

It synthesizes with the reference pipeline (the server's prompt plan, sampler and decoder) in embedding mode and
raw-reference mode, for the fine-tune and, with `--compare-base`, for the base LM. `results.json` has the cosine
similarity of each take's speaker embedding to the dataset mean. The fine-tune should beat the base and approach
`data_sim` (the real clips' own similarity). Also listen for skipped or repeated words; if that happens, use an earlier
checkpoint or a lower lr.

## 4. Export and serve

```bash
uv run --package zonos3-sgl python finetune/export.py --checkpoint runs/myvoice/best --out exports/myvoice --name myvoice
```

- `exports/myvoice/model/` is a full LM checkpoint with the LoRA merged in. Serve it with
  `SERVE_MODEL=exports/myvoice/model servers/sgl/scripts/serve.sh`, or convert it with `tools/convert_gguf.py lm` for
  the MLX, ggml and web servers.
- `exports/myvoice/voices/myvoice.npy` is the speaker embedding. Put it in the default-voices folder, or send it as
  `speaker_embedding_base64` with `"speaker_mode": "embedding"`. Raw-reference cloning with any clip of the
  speaker works too.
- `exports/myvoice/accent/myvoice.pt` (LoRA runs) is the same fine-tune as a `zonos-accent-lora/1` adapter. Serve the
  base LM with `--accent-adapters exports/myvoice/accent` and send `"accent": "myvoice"` to switch it on per request.

## Files

| File | Role |
| --- | --- |
| `ft_model.py` | `FTModel`: a differentiable forward over a loaded `zonos3.reference.lm.Zonos3LM` (same weights and names), LoRA, merge and accent export |
| `ft_data.py` | Dataset loading and the training-sequence layout |
| `prepare_data.py`, `train.py`, `sample.py`, `export.py` | The steps above |
| `test_ft_model.py` | CPU tests on a tiny random model: the forward matches the reference LM, merged weights and the accent adapter reproduce the trained forward, gradients and the sequence layout (`uv run --package zonos3-sgl python -m pytest finetune -q`) |

Notes: training runs one sequence per forward, so `--grad-accum` is the batch size. The forward matches the reference
LM to 1e-4 in fp32 (tested). bf16 adds the usual rounding. LoRA runs never change the text or audio embedding tables,
so `--phoneme-rows` and speech editing keep working on the merged model.
