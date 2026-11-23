"""CPU checks of the trainable forward against the plain-PyTorch reference on a tiny random model.

    uv run --package zonos3-sgl python -m pytest finetune/test_ft_model.py -q   (from the repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model" / "tests"))

from ft_model import FTModel, gdn_chunked, gdn_token_loop  # noqa: E402
from test_reference_lm import conditioning, prompt, tiny_model  # noqa: E402
from zonos3.reference.lm import LMCache  # noqa: E402


def _ref(model, ids, **cond):
    return model(ids, LMCache(), **cond).logits


def _ft_cond():
    c = conditioning()
    return {"speaker": c["speaker_rows"][0], "speaker_row": 0, "condition_ids": [2, 4, 0], "condition_start": 1}


@pytest.mark.parametrize("T", [1, 7, 64, 130])
def test_chunked_gdn_matches_token_loop(T):
    g = torch.Generator().manual_seed(T)
    q = torch.nn.functional.normalize(torch.randn(T, 4, 32, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(T, 4, 32, generator=g), dim=-1)
    v = torch.randn(T, 4, 32, generator=g)
    gate = -torch.nn.functional.softplus(torch.randn(T, 4, generator=g))
    beta = torch.rand(T, 4, generator=g)
    want = gdn_token_loop(q, k, v, gate, beta, 32**-0.5)
    torch.testing.assert_close(gdn_chunked(q, k, v, gate, beta, 32**-0.5, chunk=16), want, rtol=1e-4, atol=1e-4)


def test_forward_matches_reference():
    model = tiny_model()
    ft = FTModel(model, mode="lora", rank=4, gdn_backend="torch")
    ids = prompt(rows=40)
    want = _ref(model, ids, **conditioning())
    torch.testing.assert_close(ft(ids, **_ft_cond()), want, rtol=1e-4, atol=1e-4)
    ft.train()
    ft.grad_checkpoint = True
    torch.testing.assert_close(ft(ids, **_ft_cond()), want, rtol=1e-4, atol=1e-4)


def test_lora_merge_and_accent_export_match_trained_forward(tmp_path):
    from zonos3.models.accent_format import load_adapter
    from zonos3.reference.accents import AccentState

    model = tiny_model()
    ft = FTModel(model, mode="lora", rank=4, gdn_backend="torch")
    for lora in ft.loras.values():
        torch.nn.init.normal_(lora.B, std=0.05)
    ids = prompt(rows=24)
    with torch.no_grad():
        trained = ft(ids, **_ft_cond())
    assert not torch.allclose(trained, _ref(model, ids, **conditioning()), atol=1e-3)

    merged = tiny_model()
    merged.load_state_dict(ft.merged_state_dict(), strict=True)
    torch.testing.assert_close(_ref(merged, ids, **conditioning()), trained, rtol=1e-4, atol=1e-4)

    torch.save(ft.accent_adapter(), tmp_path / "voice.pt")
    state = AccentState.for_model(model, load_adapter(tmp_path / "voice.pt"), 1.0)
    got = model(ids, LMCache(), accents=state, **conditioning()).logits
    torch.testing.assert_close(got, trained, rtol=1e-4, atol=1e-4)


def test_gradients_reach_lora_and_speaker():
    model = tiny_model()
    ft = FTModel(model, mode="lora", rank=4, gdn_backend="torch")
    for lora in ft.loras.values():
        torch.nn.init.normal_(lora.B, std=0.05)
    speaker = torch.nn.Parameter(torch.linspace(-1, 1, 8))
    cond = {**_ft_cond(), "speaker": speaker}
    ft(prompt(rows=20), **cond).square().mean().backward()
    assert speaker.grad is not None and speaker.grad.abs().sum() > 0
    assert all(lora.A.grad is not None and lora.B.grad is not None for lora in ft.loras.values())
    assert not any(p.grad is not None for p in model.parameters())


def _tiny_example(model, codes, **kw):
    from ft_data import build_example
    from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig

    c = model.config
    builder = TTSPromptBuilder(TTSPromptConfig(n_codebooks=c.n_codebooks, codebook_sizes=c.codebook_sizes,
                                               audio_pad_id=c.audio_pad_id, text_vocab=448 + 8))
    # byte text ids exceed the tiny text table: remap into it
    ex = build_example(c, builder, "hi", codes, **kw)
    ex.ids[:, c.n_codebooks] = ex.ids[:, c.n_codebooks].clamp(max=c.text_vocab)
    return ex


def test_example_layout_supervises_audio_and_every_eoa():
    from ft_data import IGNORE

    model = tiny_model()
    c = model.config
    codes = torch.randint(0, c.codebook_size, (6, c.n_codebooks))
    silence = torch.zeros(c.n_codebooks, dtype=torch.long)
    ex = _tiny_example(model, codes, silence_frame=silence, silence_frames=2, with_speaker_row=True,
                       conditions=[1, 2, 3])
    prompt_len = 1 + 3 + 4  # speaker, 3 conditions, BOS h i EOS
    assert ex.speaker_row == 0 and ex.condition_start == 1
    assert len(ex.ids) == prompt_len + 2 + 6 + 1 + c.n_codebooks - 1
    sup = ex.targets != IGNORE
    assert not sup[: prompt_len + 2 - 1].any()  # nothing before the first generated row
    assert (ex.targets == c.eoa_id).sum(0).tolist() == [1] * c.n_codebooks
    # cb0 of the first generated row is the first real frame
    assert ex.targets[prompt_len + 1, 0] == codes[0, 0]


def test_lora_training_reduces_loss():
    import torch.nn.functional as F
    from ft_data import IGNORE

    model = tiny_model()
    c = model.config
    ft = FTModel(model, mode="lora", rank=8, gdn_backend="torch")
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, c.codebook_size, (12, c.n_codebooks), generator=g)
    ex = _tiny_example(model, codes, silence_frame=None, silence_frames=0, with_speaker_row=True,
                       reference_codes=None, conditions=[4, 4, 4])
    spk = torch.nn.Parameter(torch.zeros(8))
    opt = torch.optim.AdamW([*ft.parameters(), spk], lr=3e-3)

    def loss():
        logits = ft(ex.ids, speaker=spk, condition_ids=ex.condition_ids, condition_start=ex.condition_start)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ex.targets.reshape(-1), ignore_index=IGNORE)

    first = loss().item()
    for _ in range(30):
        opt.zero_grad()
        loss().backward()
        opt.step()
    assert loss().item() < 0.7 * first
