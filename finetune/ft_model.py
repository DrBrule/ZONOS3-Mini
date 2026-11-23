"""Trainable ZONOS3 Mini LM for speaker fine-tuning.

``zonos3.reference.lm.Zonos3LM`` is the numerical oracle of the release LM, but it runs under ``no_grad`` and loops the
GDN recurrence and the MoE experts token by token. ``FTModel`` wraps a loaded ``Zonos3LM`` (same parameters, same
names) with a differentiable single-sequence forward:

- global attention: SDPA with a causal mask (one sequence per step, positions from 0);
- GDN layers: FLA's ``chunk_gated_delta_rule`` on CUDA when installed (it ships with the sgl server), else a
  chunkwise pure-PyTorch gated delta rule (same math, verified against the reference's token loop in
  ``test_ft_model.py``);
- MoE: the reference's routing (top-k on ``logits - balancing_biases``, softmax over the chosen logits), experts
  gathered per expert instead of per token;
- LoRA on the accent-adapter target modules (``zonos3.models.accent_format.TARGETS``), so a trained adapter can be
  merged into ``model.pth`` or exported as a ``zonos-accent-lora/1`` adapter the servers load with
  ``--accent-adapters``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from zonos3.reference.lm import GDN_HEAD_DIM, GDN_HEADS, GDN_KERNEL, Zonos3LM, full_attnres_route, rope_interleaved

# LoRA targets (a subset of zonos3.models.accent_format.TARGETS; the router and the head are opt-in).
DEFAULT_TARGETS = (
    "attention.wq",
    "attention.wkv",
    "attention.wo",
    "attention.gater",
    "attention.in_proj",
    "feed_forward.w_in",
    "feed_forward.w_out",
    "feed_forward.latent_down",
    "feed_forward.latent_up",
)
ALL_TARGETS = DEFAULT_TARGETS + ("router.router_linear", "multi_output")


def _matches(name: str, targets) -> bool:
    return any(name == t or name.endswith("." + t) for t in targets)


def linear_weights(lm: Zonos3LM, targets) -> dict[str, nn.Parameter]:
    """Module name -> weight of every LoRA-able linear (2D, or the stacked 3D ``wkv`` / ``w_in``) matching ``targets``."""
    out = {}
    for name, module in lm.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, nn.Parameter) and weight.ndim >= 2 and _matches(name, targets):
            if isinstance(module, nn.Embedding):
                continue
            out[name] = weight
    return out


class LoRA(nn.Module):
    """``scale * B @ A`` on the flat ``[out, in]`` weight; A [rank, in], B [out, rank] (B starts at zero)."""

    def __init__(self, n_in: int, n_out: int, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, n_in))
        self.B = nn.Parameter(torch.zeros(n_out, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return (self.dropout(x.float()) @ self.A.T @ self.B.T) * self.scale

    def delta(self) -> torch.Tensor:
        return (self.B.float() @ self.A.float()) * self.scale


# ------------------------------------------------------------------------------------------------- gated delta rule


def gdn_token_loop(q, k, v, g, beta, scale):
    """The reference recurrence ([T, H, D] float32; g = log decay): oracle for tests."""
    H, D = q.shape[1], q.shape[2]
    state = q.new_zeros(H, D, D)  # [heads, value, key]
    outs = []
    for t in range(q.shape[0]):
        state = state * g[t].exp()[:, None, None]
        pred = (state * k[t, :, None, :]).sum(-1)
        delta = beta[t, :, None] * (v[t] - pred)
        state = state + delta[:, :, None] * k[t, :, None, :]
        outs.append((state * (q[t] * scale)[:, None, :]).sum(-1))
    return torch.stack(outs)


def gdn_chunked(q, k, v, g, beta, scale, chunk: int = 64):
    """Chunkwise gated delta rule (WY form, as FLA's naive reference) in plain differentiable PyTorch.
    q, k, v [T, H, D], g (log decay) and beta [T, H], all float32; returns [T, H, D]."""
    T, H, D = q.shape
    pad = (-T) % chunk
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
        g, beta = (F.pad(t, (0, 0, 0, pad)) for t in (g, beta))
    n = (T + pad) // chunk
    # [H, N, C, D]
    q, k, v = (t.transpose(0, 1).reshape(H, n, chunk, D) for t in (q * scale, k, v))
    g, beta = (t.transpose(0, 1).reshape(H, n, chunk) for t in (g, beta))
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    g = g.cumsum(-1)
    diff = g[..., :, None] - g[..., None, :]
    lower = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    decay = torch.where(lower, diff, torch.full_like(diff, -float("inf"))).exp()
    strict = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril(-1)
    L = torch.where(strict, (k_beta @ k.transpose(-1, -2)) * decay, torch.zeros_like(decay))
    eye = torch.eye(chunk, dtype=q.dtype, device=q.device)
    T_inv = torch.linalg.solve_triangular(eye + L, eye.expand_as(L), upper=False)
    k_cumdecay = T_inv @ (k_beta * g.exp()[..., None])
    v = T_inv @ v
    S = q.new_zeros(H, D, D)  # [heads, key, value]
    outs = []
    for i in range(n):
        q_i, k_i, v_i, g_i = q[:, i], k[:, i], v[:, i], g[:, i]
        attn = torch.where(lower, (q_i @ k_i.transpose(-1, -2)) * decay[:, i], torch.zeros_like(decay[:, i]))
        v_new = v_i - k_cumdecay[:, i] @ S
        outs.append((q_i * g_i.exp()[..., None]) @ S + attn @ v_new)
        last = g_i[:, -1]
        S = S * last.exp()[:, None, None] + (k_i * (last[:, None] - g_i).exp()[..., None]).transpose(-1, -2) @ v_new
    out = torch.stack(outs, 1).reshape(H, n * chunk, D).transpose(0, 1)
    return out[:T]


def gdn_fla(q, k, v, g, beta, scale):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    dtype = torch.bfloat16
    o, _ = chunk_gated_delta_rule(
        q[None].to(dtype).contiguous(),
        k[None].to(dtype).contiguous(),
        v[None].to(dtype).contiguous(),
        g[None].float().contiguous(),
        beta[None].to(dtype).contiguous(),
        scale=scale,
        output_final_state=False,
    )
    return o[0].float()


def pick_gdn_backend(name: str, device: torch.device):
    """'auto' = FLA on CUDA if it imports and matches the chunked PyTorch path on a random input, else 'torch'."""
    if name == "torch":
        return gdn_chunked
    if name in ("auto", "fla"):
        ok = device.type == "cuda"
        if ok:
            try:
                gen = torch.Generator(device="cpu").manual_seed(0)
                T, H, D = 200, GDN_HEADS, GDN_HEAD_DIM
                q, k = (F.normalize(torch.randn(T, H, D, generator=gen), dim=-1).to(device) for _ in range(2))
                v = torch.randn(T, H, D, generator=gen).to(device)
                g = -F.softplus(torch.randn(T, H, generator=gen)).to(device) * 0.3
                beta = torch.rand(T, H, generator=gen).to(device)
                want = gdn_chunked(q, k, v, g, beta, D**-0.5)
                got = gdn_fla(q, k, v, g, beta, D**-0.5)
                err = ((got - want).norm() / want.norm()).item()
                ok = err < 3e-2
                if not ok:
                    print(f"[gdn] FLA mismatch vs chunked torch (rel err {err:.3g}); using torch")
            except Exception as exc:  # noqa: BLE001
                print(f"[gdn] FLA unavailable ({type(exc).__name__}: {exc}); using torch")
                ok = False
        if ok:
            return gdn_fla
        if name == "fla":
            raise RuntimeError("--gdn-backend fla requested but FLA is not usable")
        return gdn_chunked
    raise ValueError(f"unknown GDN backend {name!r}")


# ------------------------------------------------------------------------------------------------- the model


class FTModel(nn.Module):
    """A ``Zonos3LM`` with LoRA adapters (or trainable base weights) and a differentiable forward."""

    def __init__(
        self,
        lm: Zonos3LM,
        *,
        mode: str = "lora",
        targets=DEFAULT_TARGETS,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
        train_experts: bool = False,
        gdn_backend: str = "auto",
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.lm, self.c, self.mode = lm, lm.config, mode
        self.grad_checkpoint = grad_checkpoint
        self.targets = tuple(targets)
        self.rank, self.alpha = rank, alpha
        self._gdn = pick_gdn_backend(gdn_backend, lm.device)
        self.loras = nn.ModuleDict()
        for p in lm.parameters():
            p.requires_grad_(False)
        if mode == "lora":
            for name, w in linear_weights(lm, self.targets).items():
                n_out, n_in = w[..., 0].numel(), w.shape[-1]
                self.loras[name.replace(".", "__")] = LoRA(n_in, n_out, rank, alpha, dropout).to(w.device)
        elif mode == "full":
            for name, p in lm.named_parameters():
                if name.endswith("balancing_biases") or (".experts." in name and not train_experts):
                    continue
                p.requires_grad_(True)
        else:
            raise ValueError(f"mode must be lora or full, got {mode!r}")

    # -- helpers --------------------------------------------------------------------------------------------------

    def _rms(self, x, weight=None, eps=None):
        return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=self.c.norm_eps if eps is None else eps)

    def _lin(self, name, x, weight, bias=None):
        y = F.linear(x, weight.reshape(-1, weight.shape[-1]), bias)
        lora = self.loras[name.replace(".", "__")] if name.replace(".", "__") in self.loras else None
        return y if lora is None else y + lora(x).to(y.dtype)

    # -- sublayers ------------------------------------------------------------------------------------------------

    def _attention(self, x, layer):
        c, a, p = self.c, self.lm.layers[layer].attention, f"layers.{layer}.attention."
        n, nh, nkv, hd = len(x), c.n_heads, c.n_kv_heads, c.head_dim
        gate = self._lin(p + "gater", x, a.gater.weight).sigmoid()
        q = self._lin(p + "wq", x, a.wq.weight).view(n, nh, hd)
        kv = self._lin(p + "wkv", x, a.wkv.weight)
        k, v = kv[:, : nkv * hd].reshape(n, nkv, hd), kv[:, nkv * hd :].reshape(n, nkv, hd)
        q = self._rms(q, eps=1e-6) * a.temp.abs().view(1, nh, 1).to(q.dtype)
        k = self._rms(k, eps=1e-6)
        pos = torch.arange(n, device=x.device)
        q, k = rope_interleaved(q, pos, c.rope_theta), rope_interleaved(k, pos, c.rope_theta)
        rep = nh // nkv
        key = k.to(q.dtype).repeat_interleave(rep, 1).transpose(0, 1)
        value = v.to(q.dtype).repeat_interleave(rep, 1).transpose(0, 1)
        out = F.scaled_dot_product_attention(q.transpose(0, 1), key, value, is_causal=True, scale=hd**-0.5)
        out = out.transpose(0, 1)
        return self._lin(p + "wo", (out * gate.unsqueeze(-1)).reshape(n, nh * hd), a.wo.weight)

    def _gdn_layer(self, x, layer):
        a, p = self.lm.layers[layer].attention, f"layers.{layer}.attention."
        heads, hd, kernel = GDN_HEADS, GDN_HEAD_DIM, GDN_KERNEL
        inner = heads * hd
        n = len(x)
        proj = self._lin(p + "in_proj", x, a.in_proj.weight)
        qkv, z, aa, b = proj.split((3 * inner, inner, heads, heads), -1)
        seq = F.pad(qkv.T, (kernel - 1, 0))
        conv = F.silu(F.conv1d(seq.unsqueeze(0), a.conv.weight.to(seq.dtype), groups=3 * inner).squeeze(0).T)
        q, k, v = [part.reshape(n, heads, hd).float() for part in conv.split(inner, -1)]
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        g = -a.A_log.float().exp() * F.softplus(aa.float() + a.dt_bias.float())
        beta = b.float().sigmoid()
        rec = self._gdn(q, k, v, g, beta, hd**-0.5).to(x.dtype)
        normed = self._rms(rec, a.out_weight.to(rec.dtype), eps=1e-5)
        gated = (normed.float() * F.silu(z.reshape(n, heads, hd).float())).to(x.dtype).reshape(n, inner)
        return self._lin(p + "wo", gated, a.wo.weight)

    def _ffn(self, x, layer):
        f, p = self.lm.layers[layer].feed_forward, f"layers.{layer}.feed_forward."
        if layer == 0:
            hidden = self.c.ffn_hidden
            flat = self._lin(p + "w_in", x, f.w_in.weight)
            return self._lin(p + "w_out", flat[:, :hidden] * F.silu(flat[:, hidden:]), f.w_out.weight)
        router_in = self._rms(x, f.router.rmsnorm_eda.weight.to(x.dtype))
        logits = self._lin(p + "router.router_linear", router_in, f.router.router_linear.weight)
        bias = f.router.balancing_biases.float()
        topk = self.c.moe_topk
        choices = torch.topk(logits.float().detach() - bias, topk, -1).indices
        weights = logits.float().gather(-1, choices).softmax(-1)
        latent = self._lin(p + "latent_down", x, f.latent_down.weight)
        gate_up, down = f.experts.gate_up_proj, f.experts.down_proj
        half = gate_up.shape[1] // 2
        flat_choice = choices.reshape(-1)
        tokens = torch.arange(len(x), device=x.device).repeat_interleave(topk)
        flat_w = weights.reshape(-1)
        out = torch.zeros(len(x), latent.shape[-1], dtype=torch.float32, device=x.device)
        for e in flat_choice.unique().tolist():
            sel = (flat_choice == e).nonzero(as_tuple=True)[0]
            tok = tokens[sel]
            gu = F.linear(latent[tok], gate_up[e].to(latent.dtype))
            hidden = F.silu(gu[:, :half]) * gu[:, half:]
            eo = F.linear(hidden, down[e].to(latent.dtype))
            out = out.index_add(0, tok, eo.float() * flat_w[sel, None])
        return self._lin(p + "latent_up", out.to(x.dtype), f.latent_up.weight)

    def _layer(self, layer, *sources):
        block, queries, eps = self.lm.layers[layer], self.lm.block_attnres_queries, self.c.norm_eps
        mixed = full_attnres_route(list(sources), queries[2 * layer].to(sources[0].dtype), eps)
        attn_in = self._rms(mixed, block.attention_norm.weight.to(mixed.dtype))
        attn = self._gdn_layer(attn_in, layer) if self.c.uses_gdn(layer) else self._attention(attn_in, layer)
        attn = attn.to(sources[0].dtype)  # autocast (full fine-tuning) returns bf16 next to an fp32 stream
        mixed = full_attnres_route([*sources, attn], queries[2 * layer + 1].to(attn.dtype), eps)
        ffn_in = self._rms(mixed, block.ffn_norm.weight.to(mixed.dtype))
        return attn, self._ffn(ffn_in, layer).to(attn.dtype)

    # -- forward --------------------------------------------------------------------------------------------------

    def forward(self, ids, *, speaker=None, speaker_row: int = 0, condition_ids=None, condition_start: int = -1):
        """Softcapped logits [rows, n_codebooks, audio_vocab] (float32) of one sequence ``ids`` [rows, n_codebooks + 1].
        ``speaker``: a [speaker_embedding_dim] vector replacing row ``speaker_row`` through the speaker projection;
        ``condition_ids``: one bucket id per condition table, for the rows from ``condition_start``."""
        c, lm = self.c, self.lm
        ids = ids.to(lm.device, torch.long)
        emb = lm.multi_embedder.embedders
        x = sum(F.embedding(ids[:, cb], emb[cb].weight) for cb in range(c.n_codebooks + 1))
        if speaker is not None or condition_ids is not None:
            x = x.clone()
        if speaker is not None:
            w = lm.speaker_projection
            x[speaker_row] = F.linear(speaker.to(x.device, w.weight.dtype).reshape(-1), w.weight, w.bias).to(x.dtype)
        if condition_ids is not None:
            for table, bucket in enumerate(condition_ids):
                x[condition_start + table] = lm.condition_embeddings[table].weight[int(bucket)].to(x.dtype)
        x = self._rms(x)
        sources = [x]
        for layer in range(c.n_layers):
            if self.grad_checkpoint and self.training:
                attn, ffn = checkpoint(self._layer, layer, *sources, use_reentrant=False)
            else:
                attn, ffn = self._layer(layer, *sources)
            sources += [attn, ffn]
        mixed = full_attnres_route(sources, lm.block_attnres_queries[-1].to(x.dtype), c.norm_eps)
        normed = self._rms(mixed, lm.out_norm.weight.to(mixed.dtype))
        logits = self._lin("multi_output", normed, lm.multi_output.weight).float()
        logits = logits.view(len(ids), c.n_codebooks, c.audio_vocab)
        return c.loss_softcap * torch.tanh(logits / c.loss_softcap)

    # -- adapters -------------------------------------------------------------------------------------------------

    def trainable_state(self) -> dict:
        """What a checkpoint needs: the LoRA factors (lora mode) or the trained base tensors (full mode)."""
        if self.mode == "lora":
            return {k: v.detach().cpu() for k, v in self.loras.state_dict().items()}
        return {k: v.detach().cpu() for k, v in self.lm.named_parameters() if v.requires_grad}

    def load_trainable_state(self, state: dict):
        if self.mode == "lora":
            self.loras.load_state_dict(state)
        else:
            params = dict(self.lm.named_parameters())
            for k, v in state.items():
                params[k].data.copy_(v.to(params[k].dtype))

    @torch.no_grad()
    def merged_state_dict(self, dtype=None) -> dict:
        """The base ``model.pth`` state with the adapters folded in (in each tensor's original dtype or ``dtype``)."""
        state = {k: v.detach().clone() for k, v in self.lm.state_dict().items()}
        for key, lora in self.loras.items():
            name = key.replace("__", ".") + ".weight"
            w = state[name]
            state[name] = (w.float() + lora.delta().to(w.device).reshape(w.shape)).to(w.dtype)
        if dtype is not None:
            state = {k: (v if k.endswith("balancing_biases") else v.to(dtype)) for k, v in state.items()}
        return {k: v.cpu().contiguous() for k, v in state.items()}

    @torch.no_grad()
    def accent_adapter(self) -> dict:
        """The LoRA as a ``zonos-accent-lora/1`` adapter (zonos3.models.accent_format): served at strength 1 it equals
        the merged model on every adapted linear."""
        if self.mode != "lora":
            raise ValueError("accent adapters export only LoRA training runs")
        weights = dict(self.lm.named_parameters())
        factors = {}
        for key, lora in self.loras.items():
            name = key.replace("__", ".") + ".weight"
            factors[name] = {
                "down": lora.A.detach().float().cpu().contiguous(),
                "up": (lora.B.detach().float() * lora.scale).cpu().contiguous(),
                "shape": tuple(weights[name].shape),
            }
        return {"format": "zonos-accent-lora/1", "exact": {}, "factors": factors}


