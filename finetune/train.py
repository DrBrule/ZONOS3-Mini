"""Fine-tune ZONOS3 Mini on one speaker (LoRA by default, or full fine-tuning).

    uv run --package zonos3-sgl python finetune/train.py --data data/myvoice --out runs/myvoice

Each step draws clips of the prepared dataset (prepare_data.py), lays each out as the serving prompt followed by its
delayed codec frames (ft_data.build_example) and trains next-row cross entropy over the 10 codebooks. The speaker is
conditioned through the speaker-embedding row, the raw reference prefix (another clip of the same speaker), or both
at random (``--speaker-mode mixed``, default), so either inference path works after training. Checkpoints hold the
trainable tensors and the speaker vector; export.py turns them into a merged checkpoint, an accent adapter and a
speaker .npy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ft_data import IGNORE, build_example, condition_ids, load_dataset, random_reference  # noqa: E402
from ft_model import ALL_TARGETS, DEFAULT_TARGETS, FTModel  # noqa: E402
from ft_runtime import RELEASE_LM, load_base  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="prepared dataset folder (prepare_data.py)")
    p.add_argument("--out", required=True, help="run folder (checkpoints, log.jsonl)")
    p.add_argument("--model", default=RELEASE_LM, help="base LM checkpoint dir or hf:// folder")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mode", choices=("lora", "full"), default="lora")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                   help=f"comma-separated LoRA targets (any of {', '.join(ALL_TARGETS)})")
    p.add_argument("--train-experts", action="store_true", help="full mode: also train the MoE expert weights")
    p.add_argument("--lr", type=float, default=None, help="default 1e-4 (lora) / 1e-5 (full)")
    p.add_argument("--speaker-lr", type=float, default=1e-3, help="learning rate of the learned speaker vector")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=float, default=6.0, help="used when --steps is not given")
    p.add_argument("--steps", type=int, default=None, help="optimizer steps (overrides --epochs)")
    p.add_argument("--grad-accum", type=int, default=8, help="clips per optimizer step (one clip per forward)")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-checkpoint", action="store_true", help="recompute layers in backward (less memory)")
    p.add_argument("--speaker-mode", choices=("embedding", "reference", "mixed"), default="mixed")
    p.add_argument("--speaker-vector", choices=("learned", "mean", "clip"), default="learned",
                   help="embedding mode: a trainable vector initialized at the dataset mean (exported as speaker.npy), "
                        "the frozen mean, or each clip's own embedding")
    p.add_argument("--reference-seconds", default="3,10", help="raw reference prefix length range (seconds)")
    p.add_argument("--conditions", default="loudness,speaking_rate",
                   help="condition rows filled from measured values (the rest stay 'unknown'); '' = all unknown")
    p.add_argument("--condition-dropout", type=float, default=0.3, help="chance a measured condition is set unknown")
    p.add_argument("--silence-prefill", type=int, default=3, help="codec silence frames before the audio (server: 3)")
    p.add_argument("--gdn-backend", choices=("auto", "fla", "torch"), default="auto")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--sample-every", type=int, default=0, help="write sample WAVs every N steps (0 = only at the end)")
    p.add_argument("--sample-text", action="append", default=[], help="sample sentence (repeatable)")
    p.add_argument("--language", default=None, help="language of samples (default: the dataset's)")
    p.add_argument("--resume", default=None, help="checkpoint folder to continue from")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.rng = random.Random(args.seed)
        torch.manual_seed(args.seed)
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)

        meta = json.loads((Path(args.data) / "dataset.json").read_text())
        self.language = args.language or meta.get("language", "en_us")
        self.clips, mean = load_dataset(args.data)
        self.train = [c for c in self.clips if c.split == "train"]
        self.val = [c for c in self.clips if c.split == "val"]
        if not self.train:
            raise SystemExit("no training clips in the dataset")

        dtype = torch.bfloat16 if (args.mode == "lora" and self.device.type == "cuda") else torch.float32
        self.lm = load_base(args.model, self.device, dtype)
        c = self.config = self.lm.config
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
        self.ft = FTModel(self.lm, mode=args.mode, targets=targets, rank=args.rank, alpha=args.alpha,
                          dropout=args.lora_dropout, train_experts=args.train_experts,
                          gdn_backend=args.gdn_backend, grad_checkpoint=args.grad_checkpoint)
        self.autocast = self.device.type == "cuda" and args.mode == "full"

        from zonos3.reference.synth import prompt_builder, silence_frame

        self.builder = prompt_builder(self.lm)
        self.silence = silence_frame(c.audio_codec, self.device).cpu() if args.silence_prefill > 0 else None
        from zonos3.tokenizer.matched_codec import entry

        self.frame_rate = 44100 / int(entry(c.audio_codec)["hop"])
        self.ref_max_frames = int(c.speaker_reference_max_seconds * self.frame_rate)
        self.ref_seconds = tuple(float(v) for v in args.reference_seconds.split(","))
        if args.speaker_mode != "embedding" and not c.speaker_reference_prefix_enabled:
            print("[train] this checkpoint has no raw reference prefix; using --speaker-mode embedding")
            args.speaker_mode = "embedding"
        self.use_conditions = {s.strip() for s in args.conditions.split(",") if s.strip()}

        self.speaker = torch.nn.Parameter(torch.tensor(mean, dtype=torch.float32, device=self.device),
                                          requires_grad=args.speaker_vector == "learned")
        groups = [{"params": [p for p in self.ft.parameters() if p.requires_grad],
                   "lr": args.lr or (1e-4 if args.mode == "lora" else 1e-5), "weight_decay": args.weight_decay}]
        if args.speaker_vector == "learned":
            groups.append({"params": [self.speaker], "lr": args.speaker_lr, "weight_decay": 0.0})
        for g in groups:
            g["base_lr"] = g["lr"]
        self.opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)
        n_train = sum(p.numel() for g in groups for p in g["params"])
        print(f"[train] {len(self.train)} train / {len(self.val)} val clips, {n_train / 1e6:.2f}M trainable params")

        self.step = 0
        if args.resume:
            ckpt = torch.load(Path(args.resume) / "adapter.pt", map_location="cpu", weights_only=False)
            self.ft.load_trainable_state(ckpt["trainable"])
            self.speaker.data.copy_(torch.as_tensor(ckpt["speaker"]))
            if "optimizer" in ckpt:
                self.opt.load_state_dict(ckpt["optimizer"])
            self.step = int(ckpt["step"])
            print(f"[train] resumed from {args.resume} at step {self.step}")
        self.total = args.steps or max(1, int(args.epochs * len(self.train) / args.grad_accum))
        self._order: list = []

    # -- data --------------------------------------------------------------------------------------------------------

    def next_clip(self):
        if not self._order:
            self._order = list(self.train)
            self.rng.shuffle(self._order)
        return self._order.pop()

    def example(self, clip, *, mode: str, train: bool):
        a, c = self.args, self.config
        rng = self.rng if train else random.Random(zlib.crc32(clip.id.encode()))
        reference = None
        if mode == "reference":
            reference = random_reference(self.clips, clip.id, rng, self.ref_seconds, self.frame_rate,
                                         self.ref_max_frames)
            if reference is None:
                mode = "embedding"
        conds = condition_ids(c, clip.conditions, self.use_conditions, a.condition_dropout if train else 0.0, rng)
        ex = build_example(c, self.builder, clip.text, clip.codes(), silence_frame=self.silence,
                           silence_frames=a.silence_prefill, with_speaker_row=mode == "embedding",
                           reference_codes=reference, conditions=conds)
        if len(ex.ids) > c.max_seqlen:
            return None, None
        if mode != "embedding":
            speaker = None
        elif a.speaker_vector == "clip":
            speaker = torch.as_tensor(clip.embedding, dtype=torch.float32, device=self.device)
        else:
            speaker = self.speaker
        return ex, speaker

    def pick_mode(self) -> str:
        m = self.args.speaker_mode
        return m if m != "mixed" else self.rng.choice(("embedding", "reference"))

    def loss(self, ex, speaker):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            logits = self.ft(ex.ids, speaker=speaker, speaker_row=ex.speaker_row, condition_ids=ex.condition_ids,
                             condition_start=ex.condition_start)
        targets = ex.targets.to(logits.device)
        per = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE,
                              reduction="none").view(targets.shape)
        valid = targets != IGNORE
        loss = per.sum() / valid.sum().clamp(min=1)
        cb0 = per[:, 0].sum() / valid[:, 0].sum().clamp(min=1)
        return loss, cb0.detach()

    # -- loop --------------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self):
        if not self.val:
            return {}
        self.ft.eval()
        out = {}
        modes = ["embedding"] + (["reference"] if self.args.speaker_mode != "embedding" else [])
        for mode in modes:
            losses = []
            for clip in self.val:
                ex, spk = self.example(clip, mode=mode, train=False)
                if ex is not None:
                    losses.append(self.loss(ex, spk)[0].item())
            if losses:
                out[f"val_loss_{mode}"] = float(np.mean(losses))
        self.ft.train()
        return out

    def save(self, name: str, *, optimizer: bool = True):
        folder = self.out / name
        folder.mkdir(parents=True, exist_ok=True)
        a = self.args
        ckpt = {
            "config": {"base": a.model, "mode": a.mode, "targets": list(self.ft.targets), "rank": a.rank,
                       "alpha": a.alpha, "language": self.language},
            "trainable": self.ft.trainable_state(),
            "speaker": self.speaker.detach().float().cpu().numpy(),
            "step": self.step,
            "args": vars(a),
        }
        if optimizer:
            ckpt["optimizer"] = self.opt.state_dict()
        torch.save(ckpt, folder / "adapter.pt")
        np.save(folder / "speaker.npy", ckpt["speaker"])
        return folder

    def sample(self, folder: Path):
        texts = self.args.sample_text or [
            "The quick brown fox jumps over the lazy dog, and then it naps in the warm afternoon sun.",
            "Honestly, I wasn't expecting that at all. Could you say it one more time, a little slower?",
        ]
        from sample import generate_samples

        self.ft.eval()
        try:
            generate_samples(folder, texts, self.language, base=self.args.model, ft=self.ft,
                             speaker=self.speaker.detach().float().cpu().numpy(), device=self.device,
                             reference=self.val[0] if self.val else self.train[0], data=self.args.data)
        except Exception as exc:  # noqa: BLE001 - a failed sample must not stop training
            print(f"[train] sampling failed: {type(exc).__name__}: {exc}")
        self.ft.train()

    def run(self):
        a = self.args
        log = (self.out / "log.jsonl").open("a")
        (self.out / "args.json").write_text(json.dumps(vars(a), indent=2))
        self.ft.train()
        best = float("inf")
        t0, acc_loss, acc_cb0, acc_n = time.time(), 0.0, 0.0, 0
        while self.step < self.total:
            for g in self.opt.param_groups:
                g["lr"] = lr_at(self.step, self.total, a.warmup, g["base_lr"])
            self.opt.zero_grad(set_to_none=True)
            done = 0
            while done < a.grad_accum:
                ex, spk = self.example(self.next_clip(), mode=self.pick_mode(), train=True)
                if ex is None:
                    continue
                loss, cb0 = self.loss(ex, spk)
                (loss / a.grad_accum).backward()
                acc_loss, acc_cb0, acc_n, done = acc_loss + loss.item(), acc_cb0 + cb0.item(), acc_n + 1, done + 1
            params = [p for g in self.opt.param_groups for p in g["params"]]
            gnorm = torch.nn.utils.clip_grad_norm_(params, a.grad_clip).item()
            self.opt.step()
            self.step += 1
            if self.step % a.log_every == 0 or self.step == self.total:
                rec = {"step": self.step, "loss": acc_loss / acc_n, "loss_cb0": acc_cb0 / acc_n, "grad_norm": gnorm,
                       "lr": self.opt.param_groups[0]["lr"], "sec_per_step": (time.time() - t0) / a.log_every}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                t0, acc_loss, acc_cb0, acc_n = time.time(), 0.0, 0.0, 0
            if a.eval_every and (self.step % a.eval_every == 0 or self.step == self.total):
                ev = self.evaluate()
                if ev:
                    ev["step"] = self.step
                    print(json.dumps(ev), flush=True)
                    log.write(json.dumps(ev) + "\n")
                    log.flush()
                    score = ev.get("val_loss_embedding", float("inf"))
                    if score < best:
                        best = score
                        self.save("best", optimizer=False)
            if a.save_every and self.step % a.save_every == 0:
                self.save(f"step-{self.step}")
            if a.sample_every and self.step % a.sample_every == 0:
                self.sample(self.save(f"step-{self.step}"))
        final = self.save("final")
        self.sample(final)
        print(f"[train] done: {final} (best val checkpoint: {self.out / 'best'})")


def main(argv=None):
    Trainer(parse_args(argv)).run()


if __name__ == "__main__":
    main()
