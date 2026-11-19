# TTS System Architecture

## Overview

mini-sglang serves a multi-codebook TTS transformer (Zonos3) through a multi-process pipeline with streaming audio output. Text is encoded as SuperBPE (or UTF-8 byte) token frames, fed through an autoregressive transformer that predicts 10 parallel audio codebook streams of the matched dual codec, and decoded to 44.1 kHz PCM audio by that codec's causal iSTFT decoder - with streaming responses to the HTTP client.

Paths below are relative to the two packages: `zonos3/` (`model/src/zonos3`, model code shared with the ggml server and the reference) and `zonos3_sgl/` (`servers/sgl/src/zonos3_sgl`, the CUDA server).

## Process Architecture

Four process types communicate over ZMQ:

```
                  ZMQ                ZMQ                ZMQ
  Frontend  ──────────►  Tokenizer  ──────────►  Scheduler  ──────────►  Detokenizer
  (FastAPI)  ◄──────────             ◄──────────  (Engine)   ◄──────────  (Vocoder)
             TTSAudioReply          TTSUserMsg             TTSDetokenizeMsg
```

| Process | Count | Role |
|---------|-------|------|
| Frontend | 1 | FastAPI server. Accepts `POST /v2/generate` and `POST /v1/audio/speech`, streams PCM chunks back. Logs TTFB, E2E latency, RTF. |
| Tokenizer | N (default 0 = shared with detokenizer) | Runs `TTSPromptBuilder`: text → SuperBPE / byte tokens → 2D prompt tensor (speaker row, condition rows, reference prefix, text). Sends `TTSUserMsg` to scheduler. |
| Scheduler | 1 per TP rank | Runs `TTSScheduler` wrapping `Engine`. Autoregressive decode loop: prefill → decode → sample → emit frames. |
| Detokenizer | 1 | Runs `TTSVocoderManager`: audio codes → matched codec decoder → PCM bytes. Sends `TTSAudioReply` to frontend. |

Spawned by `launch_server()` in `zonos3_sgl/server/launch.py`. Each scheduler process holds one GPU; tensor parallelism across ranks is coordinated via NCCL.

### Message types

Defined in `zonos3_sgl/message/tts.py`:

| Message | Direction | Key fields |
|---------|-----------|------------|
| `TTSTokenizeMsg` | Frontend → Tokenizer | `uid`, `text`, `sampling_params`, optional speaker/condition conditioning |
| `TTSUserMsg` | Tokenizer → Scheduler | `uid`, `input_ids: Tensor (seq_len, frame_width)`, `sampling_params` |
| `TTSDetokenizeMsg` | Scheduler → Detokenizer | `uid`, `audio_codes: List[int]` (one frame), `finished`, `eos_frame` |
| `TTSAudioReply` | Detokenizer → Frontend | `uid`, `audio_data: bytes` (float32 PCM), `finished`, `sample_rate` |

## Token Format

Each time step is a **frame** of width `n_codebooks + 1`:

```
[cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9, text_token]
 -------------- 10 audio codebooks ---------------   text id
```

- **Audio codebook tokens**: integers in `[0, 4095]` (codebook_size = 4096).
- **Special tokens**: `eoa_id = 4096` (end-of-audio), `audio_pad_id = 4097` (padding); speech-editing checkpoints add `span_end_id = 4098`.
- **Text column**: SuperBPE (or byte) token ID during prompt; `text_vocab` placeholder during generation. Extra rows after the text padding row hold the edit markers and phoneme-group tokens (`text_extra_tokens`).
- **Conditioning rows**: speaker-embedding checkpoints put a speaker row (`speaker_projection`, 192 → model dim) before the text; checkpoints with condition tokens (ZONOS3 Mini: `loudness`, `fmax`, `speaking_rate`; table i has `condition_bucket_counts[i]` buckets plus an unknown row, 10 / 10 / 30 on the default `lm/rate30_20k_r2grpo`, 10 each on `lm/mect_60k`) add one condition row per condition (`condition_embeddings`). Raw-reference cloning places the delayed reference codes (the reference prefix) before the text.

During the prompt phase, audio columns are filled with `audio_pad_id` and the text column carries the token sequence. An optional codec-matched silence prefill (`silence_prefill_frames`; default 3 frames = 0.2 s, `ZONOS3_SILENCE_PREFILL_FRAMES` / `ZONOS3_SILENCE_PREFILL_DEFAULT`) appends the delayed rows of the registry's `silence_frame` after the text; the vocoder decodes them as left context and never emits them.

### Shear pattern

The model is trained with an inter-codebook delay: codebook `j` is shifted by `j` frames. `shear()` (`zonos3/tts/prompt.py`) applies this delay to the prompt; `shear_up()` (`zonos3/tts/delay.py`) inverts it before vocoder decoding.

### Extra text rows

`tools/add_text_extra_rows.py` writes `text_extra_tokens` and `text_tokenizer_spec.phonemes` (group markers,
token ids, symbols), `start` and `pauses` into `params.json`. `text_extra_tokens` is the same field the speech-edit
vocabulary uses for its marker rows. The phoneme rows, then the start row, then the pause rows are always the last
extra rows: right after the padding row, or after the edit marker rows (which stay at text_vocab + 1 + j, where edit
training put them); `superbpe.load_spec` checks that markers plus these rows fill `text_extra_tokens`, and
`edit_max_spans` counts the marker rows. With the speech-edit vocabulary the result has `text_extra_tokens` 146 (3
markers, open / close 8200 / 8201, phones from 8202, start 8338, pauses 8339-8342), `speech_editing` with
`edit_max_spans` 3 and the 136 phone symbols.

## Request Lifecycle

```
1. HTTP POST /v2/generate (or /v1/audio/speech)
   ├─ FrontendManager sends TTSNormalizeMsg(text, language) first: the tokenizer normalizes the text (the Rust
   │  zyphra_textnorm extension) while the frontend resolves the request's rules and speaker
   └─ FrontendManager assigns uid, creates TTSTokenizeMsg

2. Tokenizer process
   ├─ text normalization                   # TTSTextNormalizer, memoized per (text, language): a hit after step 1
   ├─ TTSPromptBuilder.build(text)         # SuperBPE / byte tokenization (zonos3/tts/prompt.py)
   │   └─ per text token: [audio_pad_id]*10 + [text_token_id]
   ├─ speaker row / condition rows / reference prefix before the text
   ├─ append sheared silence prefill rows  # optional, codec-matched silence frame (default 3 frames)
   └─ emit TTSUserMsg(input_ids: 2D tensor)

3. Scheduler
   ├─ Prefill: engine.forward_batch_tts()  # full prompt in one pass (GDN checkpoints: a CUDA graph at padded shapes)
   │   └─ extract last-token logits per sequence
   └─ Decode loop (frame by frame):
       ├─ engine.forward_batch_tts()       # single frame input
       ├─ sample_tts(logits)               # → [cb0..cb9, text_placeholder] on the GPU, without a host sync
       │                                   #   (the host schedules the next step while the GPU runs this one; the
       │                                   #   scheduler stream waits on the engine stream before touching the token pool)
       ├─ TTSSequence.append_token()       # EOS detection
       ├─ emit TTSDetokenizeMsg per frame
       └─ repeat until is_finished

4. Detokenizer (TTSVocoderManager)
   ├─ accumulate frames per uid
   ├─ shear_up() to remove delay pattern
   ├─ matched codec decode (iSTFT head) → float32 PCM @ 44.1 kHz
   │   (whole utterance at EOS, or progressive prefix chunks with stream_decode="prefix")
   └─ emit TTSAudioReply

5. Frontend streams audio_data bytes to HTTP client
```

## Model Architecture (Zonos3)

`Zonos3ForCausalLM` in `zonos3_sgl/models/zonos3.py` (the plain-PyTorch twin with the same parameter names is `Zonos3LM` in `zonos3/reference/lm.py`).

```
Input (seq_len, frame_width)
  │
  ▼
MultiEmbedding              # sum of per-column embeddings → (seq_len, hidden_size)
  │
  ▼
emb_norm                    # RMSNorm (no learnable params)
  │
  ▼
TransformerBlock × N        # residual stream with fused RMSNorm
  │  ├─ attention_norm (RMSNormFused)
  │  ├─ Attention
  │  │   ├─ wq, wkv, wo projections
  │  │   ├─ QK norm + learnable temperature (per-head)
  │  │   ├─ RoPE (interleaved format)
  │  │   ├─ FlashAttention / FlashInfer backend
  │  │   └─ headwise gating (sigmoid)
  │  ├─ ffn_norm (RMSNormFused)
  │  ├─ FeedForward (dense) or MoEFeedForward
  │
  ▼
out_norm (RMSNormFused)
  │
  ▼
MultiOutputHead             # linear: hidden → (n_codebooks × audio_vocab)
  │                         #   reshape → (*, 10, 4098) for ZONOS3 Mini
  ▼
softcap(logits, loss_softcap)  # tanh-based logit capping (15.0 for ZONOS3 Mini)
```

### MultiEmbedding

Maintains separate `VocabParallelEmbedding` tables for each of the `n_codebooks` audio codebooks (10 for ZONOS3 Mini) plus the text column. Forward pass sums all column embeddings element-wise.

### Attention

- **QK normalization**: `F.rms_norm` on Q and K, then Q is scaled by `|temp|` (learnable per-head parameter).
- **Gating**: After attention output, applies `sigmoid(gater(x))` either per-head (`headwise`) or per-element (`elementwise`), then multiplies with the attention result.
- **RoPE**: Interleaved format (`is_neox=False`).

### MoE

Activated for middle layers (`moe_start_from_layer` to `num_layers - moe_end_from_layer`).

- **Router**: Projects hidden → `router_dim` via `down_proj`, then a 3-layer MLP with GELUs produces expert logits.
- **Expert Dropout Augmentation (EDA)**: Non-first MoE layers blend the previous layer's router hidden states (`router_states * scale + current`) before routing. This threads router context across consecutive MoE layers.
- **Experts**: `FusedGroupedExperts` using fused gate+up projection (`w1 || w3` → `gate_up_proj`) with SiLU activation, and a down projection (`w2` → `down_proj`). Uses the `FusedMoE` kernel for efficient dispatch.

## Sampling

`sample_tts()` in `zonos3/tts/sampler.py`. Input: `(B, n_codebooks, vocab_size)` logits.

1. **Repetition penalty**: For each request and selected codebook, gather the last `repetition_window` generated tokens from that codebook only and apply `repetition_penalty` to matching logits. `repetition_codebooks=N` applies this to CB0 through CB(N-1), while a negative value means all codebooks. `repetition_window=0` or `repetition_penalty=1.0` disables this path.
2. **Temperature**: Per-sequence scaling, broadcast over codebooks.
3. **Top-k**: Single `torch.topk` with `min(top_ks)` across the batch.
4. **Top-p / Min-p**: Applied to the softmax probabilities using the most restrictive value across the batch; `top_p=0` and `min_p=0` disable them.
5. **Multinomial sampling**: Per-codebook independently. Supports per-request `torch.Generator` for seeded determinism.
6. **Output**: `[cb0, ..., cb9, text_vocab]` — audio tokens plus text placeholder.

Default params (`TTSSamplingParams`): temperature=0.75, topk=80, top_p=0.69, min_p=0.0, max_tokens=1024, repetition_window=96, repetition_penalty=2.4, repetition_codebooks=6.

## EOS Detection

Managed by `TTSReq.check_eos()` in `zonos3_sgl/core.py`. Each request has one stop id (`TTSReq.stop_id`): EOA for utterances
(`end_token="utterance"`), the span-end token for edit fills (`end_token="span"`, edit checkpoints only).

1. Each frame, check whether any delayed audio codebook sampled the request's stop id.
2. When detected: align `eos_frame` by subtracting the highest stop-token codebook index, then start countdown = `n_codebooks + 1` (11 steps for 10 codebooks). `--eoa-policy codebook0` lets only a stop token in codebook 0 start it (default `any`).
3. Sequence marked finished when countdown reaches 0 — this buffer allows frame alignment across codebooks due to the shear pattern.
4. Also terminates if `num_completion_tokens >= max_tokens`.

The sampler (`sample_tts` in `zonos3/tts/sampler.py`) masks padding and every special id except the request's stop id, so an
utterance never samples span-end and a fill never samples EOA. Utterances draw over codes, EOA and padding only: with
one random number per column in `torch.multinomial`, an edit checkpoint then produces the same audio for the same
seed as its weights without the span-end row.

Stop boundaries count host-processed frames, not submitted GPU work.

## Vocoder (matched dual codec)

`TTSVocoderManager` in `zonos3_sgl/tokenizer/vocoder.py` decodes with the codec the LM's training data was tokenized with (the
checkpoint's `audio_codec`). Only matched codecs of the codec registry are accepted. Each request decodes through the
decoder it names (`decoder`, a served decoder id; null = the default `keep2_plain`), carried on its sampling params to
every `TTSDetokenizeMsg` and on `TTSDecodeCodesMsg` (edits).

### Codec registry

`zonos3/registries/codec_registry.json` holds one entry per codec. The ZONOS3 Mini entry
(`opus_dual_3n_600k_sched600k_r4096_step600000`, family `dual`) declares:

- `codebooks` 10, `codebook_sizes` 10 × 4096, `hop` 2940 samples at 44.1 kHz (`frame_rate` 15), `codec_step`;
- `checkpoint` + `checkpoint_sha256`: the codec generator (`hf://…/autoencoder/generator.pt`), whose sha256 is checked
  before the first decode;
- `teacher`: the semantic SSL teacher the encoder needs (`hf://…/autoencoder/teacher`);
- `silence_frame`: the steady-state codes of digital silence, used for the silence prefill;
- `decoders`: the decoders a request may pick by id, and `decoder_default` (`keep2_plain`). Each is `kind:
  "istft_mr"` with its `campaign` folder, `config` and `checkpoint`, `expect_step`, and `headroom_db` (3.0: the default
  output gain of −3 dB, overridable per request with `headroom_db`; the iSTFT head is unbounded):
  - `keep2_plain` (the default): the KEEP-2 pruned head (`autoencoder/heads/keep2_plain`,
    `config_full_unfreezeall_mr_drop30_4x10.json`: two layers per transformer stage);
  - `full_mr`: the full-size head (`autoencoder/heads/full_mr`, `config_full_unfreezeall_mr.json`, decoder.0-8
    unfrozen).

  Both share the codec, its frame rate, codebooks and the stream / flush semantics; they differ only in layer counts.
  The GGUF servers serve the same two heads as `zac-keep2.gguf` and `zac.gguf` (the id is the GGUF's
  `mossdual_istft.head.variant`), the web engine as its exported model directories.

### Decoder

`zonos3.tokenizer.matched_codec.load_model` builds `zonos3.codec.istft_head_a_20261002.model.ISTFTCodec` from the
decoder config, loads the generator's pretrained weights (`generator.pt`: quantizer and the initial decoder stages),
then the MR head's trainable state (decoder stages + causal multi-resolution iSTFT head) from the decoder checkpoint;
the generator's conv-tail decoder is never used. The model is cached per decoder id and device
(`ZONOS3_AUDIO_CODEC_DEVICE`, default `cuda:0`); `decode`, `stream`, `capture_stream_graphs`, `decoder_info` and
`decoder_headroom_db` take the decoder id (None = the default). `decode_model` runs `[B, frames, 10]` codes in fp32 with
TF32 off and checks the output length (`frames × hop`) and finiteness.

The sgl server loads every decoder of the registry entry at startup (`--decoders keep2_plain,full_mr` restricts the
set; the registry default stays the default when listed, else the first listed is); `GET /v2/capabilities` lists them
in `decoders` (default first, also `decoder`), a request picks one with `decoder` (unknown id: 400) and audio
responses name it in `X-Decoder`. Everything below runs per decoder exactly as for the default.

### Streaming decode

- Frames accumulate in a per-request buffer (after any silence prefill rows, which are decoded but never emitted).
- With `N` accumulated rows, only `N - (n_codebooks - 1)` output frames are complete (the rest need later rows for
  `shear_up`); the EOS frame (from the scheduler's `eos_frame` or an EOA seen locally in a row) bounds them further.
- **Whole decode** (default for non-streaming responses, and `stream_decode="whole"`): at EOS, `_decode_final`
  undelays the rows with `shear_up` and decodes every complete pre-EOS frame in one codec call. The last
  `overlap_frames` (default 4) already-counted rows (the silence prefill) are re-decoded as left context.
- **Prefix decode** (`stream_decode="prefix"`, the server default for streaming responses;
  `ZONOS3_STREAM_DECODE_DEFAULT=whole` turns it off): the frames are decoded incrementally on a stateful codec stream
  (conv histories, per-layer KV caches, overlap-add iSTFT, FIR mixer and resampler state;
  `zonos3.tts.stream.PrefixDecoder` over `ISTFTStream`), nothing decoded twice. Chunks ramp: the first waits
  for `ZONOS3_PREFIX_STREAM_FIRST_CHUNK_FRAMES` (1) complete frames past the silence prefill, each later one for
  `first + frames emitted so far`, capped at `ZONOS3_PREFIX_STREAM_CHUNK_FRAMES` (15, 1 s): 1, 2, 4, 8, 15, 15, ...
  `ZONOS3_PREFIX_STREAM_HOLD_BACK_FRAMES` (0) complete frames stay undecoded until EOS, so first audio needs
  9 (shear) + 1 generated frames. At EOS the stream continues to the end and is flushed. The stream emits only final
  samples, so the concatenated chunks equal the whole decode in fp32 (CPU); on CUDA the trunk runs in bf16 and any
  change of chunk boundaries moves the PCM by ~1e-3 RMS. `ZONOS3_PREFIX_STREAM_CONTEXT_FRAMES` is still validated
  (>= chunk) but unused.
- **Primed stream** (prefix decode with the silence prefill rows and no decode context): the vocoder decodes the
  prefill frames once as pure silence and every such request continues a fork of that stream
  (`TTSVocoderManager._prefix_stream`, one primed stream per decoder; `ISTFTStream.fork`: the trunk state tensors are
  shared, never written in place; the head's overlap-add buffers are copied), so the first chunk is the first
  generated frame alone. The
  prefill frames' codebooks that fall in generated rows (frame f, codebook j with f + j >= prefill) are the LM's and
  are not decoded: the PCM moves ~1.7e-3 RMS (signal ~0.16), the same size as moving a chunk boundary
  (~1.4e-3); the codes are unchanged.
- **Codec CUDA graphs**: at tokenizer start `warm_prefix_stream` primes, for every served decoder, the default prefill's stream and captures the
  chunk shapes of the primed ramp (3, then 1, 2, 4, 8, 15 ... until every KV cache is full at 150 frames, then the
  steady 15-frame chunk; `zonos3.tts.stream.prefix_chunks`, `matched_codec.capture_stream_graphs`). A chunk whose
  shape (frames, stream position up to `ISTFTStream.full_frames`) was captured replays `ChunkGraphs` (state copied
  in and out, position as a graph input); other shapes (decode context, other prefills, EOS tails) run eagerly.
  Replays are bit-identical to the eager decode; they cut the GPU half of the first chunk from ~11.8 ms of launches
  to ~2.9 ms.
- **Device iSTFT head**: the overlap-add iSTFT, FIR mixer and resampler of a stream run on the codec's device
  (`zonos3.codec.stream_device`, float64 like the NumPy ports in `zonos3.codec.stream_host` that the MLX server keeps,
  and bit-identical to them), so a chunk's only host copy is its final samples. On one H100 the first chunk's
  `decode_frames` falls from ~6.8 to ~4.3 ms (iSTFT + mixer 2.25 -> 0.60 ms, resampler 0.30 -> 0.19 ms, primed-stream
  fork 0.60 -> 0.06 ms), leaving the trunk's graph replay (~3.1 ms) as the cost.
- **Early partial-frame decode** (on by default in every server: `stream_codebooks` = 1 on the `first` chunk, the
  generation registry's `stream_codebooks`; k = n_codebooks waits for complete frames): a frame is decodable once its
  first k codebooks arrived (row f + k - 1 instead of f + 9; `zonos3.tts.stream.PrefixDecoder.ready`, capped at EOS
  and before any frame whose semantic code is padding/EOA). The missing residual codes reach the codec as -1 and the
  codec leaves those codebooks out of the RVQ sum, bias included (the codec trained with quantizer dropout 0.5; the
  GGUF codecs need the per-codebook `res.{i}.bias` tensors, without them early decode is off for that model); a frame
  is decoded once, never refined. With scope `first` (default) only the first chunk is early and later chunks wait
  for complete frames; `all` decodes every chunk early. The codes are unchanged; the complete-code decode stays
  bit-identical. Measured on sgl / one H100 (`early_sweep.py`): TTFA ~52.5 ms at k = 10 falls ~3.1 ms per codebook to
  ~24.4 ms at k = 1; against the k = 10 take of the same seed the waveform SNR is 23.9 dB with scope `first` (6.5 dB
  with `all`).
- With a raw-reference prompt, `decode_context_frames` (default 30) of the reference's tail codes are prepended as
  decoder context whenever a decode starts at the stream start, and their samples dropped.

### Decode pipeline

```
delayed rows (seq_len, 10)
  → shear_up(): remove inter-codebook delay, fill with audio_pad_id
  → slice to the complete pre-EOS frames
  → matched_codec.decode(codes, codec, decoder): sha256 check, ISTFTCodec of that decoder (fp32, TF32 off)
  → waveform (float32, 44.1 kHz, frames × 2940 samples)
```

### Reference encoder

Raw-reference cloning and the silence-frame fallback encode audio with the same codec:
`zonos3/tokenizer/dual_reference.py` takes the registry's single `dual`-family entry and builds
`zonos3.codec.codec_adapter.MossAutoencoder` from its generator checkpoint, SSL teacher, step and codebook sizes
(SSL teacher → semantic Q1, acoustic residual RVQ). Reference audio is mixed to mono, resampled to 44.1 kHz, peak
normalized only above 1.0 and cut to whole codec frames within the reference limit; the result is int32 codes
`[frames, 10]`. The sgl server encodes in its API process, on one worker thread, and memoizes the codes by the clip's
bytes; the tokenizer processes only receive codes. The encoder holds several GB of GPU memory per process that loads
it, so `--num-tokenizer N` must not give each tokenizer process its own.

## GDN / Full-AttnRes family (2026-10)

Checkpoints with `attention_pattern != "global"` use `zonos3_sgl/models/gdn.py` (`OriginalGDN`): the canonical 64-frame chunked
gated-delta-rule recompute (FLA's `chunk_gated_delta_rule` + `causal_conv1d` forward functions, called directly with
`cu_seqlens` and chunk indices built on the host and copied without blocking; FLA would derive the indices with a host
sync per call) with per-slot float32 recurrent and
convolution states (`--cache-type naive`). The unfinished chunk of every slot is kept in a fixed 64-row window
(`_window`, `_win_len`), which the eager prefill path reads and writes and which makes decode a static-shape
computation: `_forward_decode_static` processes every padded request as one 64-row sequence (real rows first, zero
rows after; the kernels are causal, so the real rows are bit-identical to the ragged call), gathers the new row's
output and commits the states only when a window fills. `Engine` captures that step in CUDA graphs per batch size
(`GraphRunner` hooks `before_capture` / `after_capture` / `before_replay` fill the static slot buffer and switch the
layers' static mode; `note_graph_forward` keeps the host mirrors current after a replay). Two details keep replays
valid: all layers share one set of static `cu_seqlens` / chunk-index tensors (FLA memoizes results derived from them by
tensor identity in 4-entry caches), and the engine pins the cache results those tensors produced (`pin_fla_cache`) because
the captured kernels hold raw pointers to them and eager prefills would otherwise evict and free them. The fused
Full-AttnRes bank is one persistent context per graphed shape (`_attnres_state`).

Prefills of fresh requests run as CUDA graphs too (`zonos3_sgl/engine/prefill_graph.py`, `PrefillGraphs`): eagerly a
prefill launches ~900 kernels (~1,400 with a reference prefix) for 3-6 ms of GPU work in 40-60 ms of host time. Each
request is padded to its complete 64-row chunks plus one 64-row window (`PrefillLayout`), so the layer stack's shapes
depend only on the requests' complete-chunk counts; one graph per tuple of counts, captured on first use (single
requests of up to 5 complete chunks at startup) and kept in an LRU (`ZONOS3_PREFILL_GRAPHS`, 16) for prefills of up to
`ZONOS3_PREFILL_GRAPH_MAX_ROWS` padded rows (1024); `ZONOS3_PREFILL_CUDA_GRAPH=0` turns them off. The embedding
prologue (speaker and condition rows) runs eagerly on the real rows (`Zonos3ForCausalLM.embed`), which are scattered
into the graph's padded input; the graph runs `forward_layers`, the head and the last-row gather. No real row reads a
padded one: attention is causal over each request's own page-table row (padded positions write their KV to the dummy
page), `_forward_prefill_static` runs the eager path's committed pass over the complete chunks and its uncommitted tail
pass as a zero-padded 64-row window (the decode window's layout), and every other kernel is batch-invariant (below), so
graph and eager prefills leave bit-identical logits, GDN states and KV rows. Continuations (a cached prefix: LACI
rollbacks) and accent adapters stay eager. `residual_type == "full_attnres"` routes
every sub-layer output through `full_attnres_route()` in `zonos3_sgl/models/zonos3.py` (reference schedule
`_forward_full_attnres_reference`, or the trainer's Triton kernels in `zonos3_sgl/models/training_attnres*.py` when
`ZONOS3_USE_TRAINING_ATTNRES=1`). The `ZONOS3_USE_TRAINING_{RMSNORM,ROUTER_WEIGHTS,SONIC_EXPERTS,ATTNRES,ROPE}` switches
make the forward bit-exact with the packed training forward ("exact mode"); the scheduler samples these models from
float32 logits (`sample_logits_in_fp32` in `zonos3/tts/sampler.py`). Exact mode is also batch-invariant and exact at long
context: `MultiOutputHead` pads the head GEMM to `head_min_rows()` (2 for this family) because cuBLAS uses a different,
one-ulp-off kernel for a single row, and `FlashAttentionBackend` passes `fa_num_splits()` (1 for this family, the
training kernel's setting) because FA3's automatic KV split rounds differently for caches beyond ~512 rows.
`Engine._check_head_batch_invariance()` verifies the head property at load. Every other kernel in the exact path (all
other linears, the FLA chunk kernel, SonicMoE, the training RMSNorm and AttnRes kernels) is batch-invariant as is.

Two exact-mode kernels are Hopper-only: SonicMoE's quack grouped GEMM ("Gemm Sm80 is not implemented yet") and FLA's
TileLang chunk kernels. On a GPU below sm_90 (A100) `servers/sgl/scripts/env.sh` leaves
`ZONOS3_USE_TRAINING_SONIC_EXPERTS` and `FLA_TILELANG` off, so the experts run the Triton `FusedMoE` and the GDN chunks
FLA's Triton kernels; the rest of exact mode (RMSNorm, router weights, AttnRes, RoPE) is portable, and the `fa` backend's
sgl-kernel FA3 has an sm_80 build (head_dim 32 runs on its 64-wide kernels, as on Hopper). The forward is no longer
bit-exact with training. Forced on an H100 (`FLA_TILELANG=0 ZONOS3_USE_TRAINING_SONIC_EXPERTS=0`), the profile measured:
the expert layer 20 vs 42 us at 1 token and 97 vs 204 us at 1024 (SonicMoE is a training kernel), TTFA p50 13.7 vs 14.7 ms,
1700 CV3 / MiniMax samples at 256 running requests in 110 vs 115 s, and the same WER / SIM within seed noise (pooled word
errors -1.6 %).

Text uses the SuperBPE convention in `zonos3/tts/superbpe.py` (`[2] + (ids + 4) + [3]`, artifact pinned by SHA-256 through
`params.json`). Audio uses 10 codebooks of 4096 (`eoa_id` 4096, `audio_pad_id` 4097) from the dual codec; the matched
codec registry (`zonos3/tokenizer/matched_codec.py`) provides both the encoder for reference audio
(`zonos3/tokenizer/dual_reference.py`) and the vocoder. The raw-audio reference prefix (`speaker_reference_prefix_tokens` in
`zonos3/tts/prompt.py`, or caller-supplied `speaker_reference_codes`) precedes the text rows; speaker CFG removes it for the
unconditional twin. Speech-editing checkpoints add a span-end row per codebook (`span_end_id` 4098) and
`text_extra_tokens` marker rows after the text vocabulary; `zonos3/tts/edit.py` builds the fill prompts
([guide: Speech editing](guide.md#speech-editing)).

## Speech editing pipeline

`TTSLLM.edit(codes, target_text, spans)` generates each masked span
(`(start_frame, length_frames)`, ascending, at most the checkpoint's edit marker rows: `edit_max_spans`, 3 on the
release) as its own request over `[speaker row][condition rows][text][unmasked context][earlier
fills][marker]` and splices the fills into the untouched codec frames.

For `min_fill_ratio` ([guide: Edit request fields](guide.md#edit-request-fields)), the
sampler masks the stop id per codebook while generated frame s < `min_frames`
+ c (`TTSSamplingParams.min_frames`), so the aligned stop frame is at least
`min_frames`; requests without it build no mask and sample as before.
`max_span_seconds` stays the upper bound.

The frontend runs an edit through the process pipeline: it encodes the audio
with the server's one codec encoder (the same memoized encode as raw-reference
prompts), the tokenizer builds each fill prompt (`TTSTokenizeMsg.edit_*`), the
scheduler generates it with `end_token="span"`, the detokenizer returns its rows
as codes (`TTSCodesReply`), and the tokenizer decodes the spliced frames
(`TTSDecodeCodesMsg`).

In speech edits ([guide: Groups in speech edits](guide.md#groups-in-speech-edits)), the edit
tokens' normalization would mangle a group (IPA upper-cased, the markers and tie bars
become word breaks, so `t͡ʃiːz` reads `T ƩIːZ`), so a group is never normalized: it
is one target token of its own in the word diff, windows and expected fill length
(its length = its IPA symbols × 1.4 letter-equivalents,
`GROUP_LETTERS_PER_SYMBOL`), and window texts keep it as typed, so fill prompts
carry the group ids as `/v2/generate` encodes them. With `verify`, each group is
a wildcard word of the WER / CER reference (any one recognized word matches; ties
go to the alignment with the fewest character edits), the take's word alignment
uses the target with each group replaced by the word heard, and `verify.groups`
lists every group of `target_text` in order as `{ipa, heard}`. The 1.4 letters
per IPA symbol (rather than the lexicon's average of about one) reflects that
spoken duration follows the word, and respelled names and short words carry more
letters per phone.

## Accent adapters

A request runs base + α × adapter, the merge the adapter was scored as offline.
Every loaded adapter owns `rank` columns of one shared space, so an adapted
layer computes `x W^T + ((x D^T) * M) U` with `M` holding each token's strength
on its adapter's columns; the router balancing biases (quantile thresholds) get
each request's scaled shift per token. Per-row slots and strengths are CUDA-graph
inputs. The extra cost is fixed by the total columns loaded, not by how many
accents a batch mixes.

The adapter keys are the training code's module names, which the converter keeps, and
`zonos3.models.accent_format.TARGETS` mirrors the training code's
default targets. Every target adds its shift to the flat
`[out, in]` product before the runtime splits it, the product the training code's LoRA
adapts:

| Adapter target | served op | layout |
|---|---|---|
| `attention.wq`, `attention.gater` | `Attention.wq` / `.gater` (`SimpleLinear`) | as stored |
| `attention.wkv` | `Attention.wkv` (`ChunkedLinear`) | `[2, kv, in]` as `[2·kv, in]`, K\|V split after |
| `attention.wo` | `Attention.wo` (`SimpleLinearOProj`) or `OriginalGDN.wo` | as stored |
| `attention.in_proj` | `OriginalGDN.in_proj` | fused `[2056, in]`, qkv\|z\|a\|b split after; also applied to the replayed 64-row chunk tail and the graph's fixed windows, with the request's own accent |
| `feed_forward.w_in` / `w_out` | `FeedForward.w_in` (`ChunkedLinear`) / `.w_out` | `[2, inter, in]` as `[2·inter, in]`, h\|gate split after |
| `feed_forward.latent_down` / `latent_up` | `MoEFeedForward.latent_down` / `.latent_up` | as stored, around the experts |
| `router.router_linear` | `Router.router_linear` | logits feed the softmax and the quantile top-k |
| `multi_output` | `MultiOutputHead` | shift on the real rows after the `min_rows` padding |
| `router.balancing_biases` (exact float32 shift) | `Router` balancing bias | per-slot table; legacy ZeroMean `parametrizations.balancing_biases.original` re-centred, `ent_denom` / `normalized_entropy` dropped, as in the converter |

Any other moved tensor is refused at load.

**Expert factors (`zonos-accent-lora/2`).** Adapters trained with
`lora.expert_rank` also shift every MoE expert weight by its own low-rank
product. They export `layers.N.feed_forward.experts.w13` / `.w2` entries with
`up` `[E, out, r]`, `down` `[E, r, in]`, `shape` and `experts: E`, in the
SonicExperts layout: w13 is `[64, 2·1536, 256]` with gate/up rows interleaved,
w2 `[64, 256, 1536]`. The experts sit in the 256-wide latent between
`latent_down` and `latent_up`, top-2, on layers 1-10. The non-expert entries
and the router bias shifts are as in `/1`. Expert factors are served on the
exact SonicMoE path only (`ZONOS3_USE_TRAINING_SONIC_EXPERTS=1`, which
`servers/sgl/scripts/env.sh` sets on Hopper; SonicMoE has no A100 build); elsewhere they are refused, as are expert entries in a
`/1` file, expert weights other than w13 / w2, mismatched shapes or expert
counts, and an adapter whose w13 and w2 ranks differ.

For a routed (token, expert e) pair of a request on adapter a at strength α:

- `α · U13[a,e] (D13[a,e] x)` is added to the gate/up pre-activation, before
  SwiGLU;
- `α · U2[a,e] (D2[a,e] h)` is added to the pair's expert output, before the
  weighted top-2 sum (`h`: the adapted activation).

Training formed `W_e + U_e D_e` in the forward, so this is the same function
with the sum regrouped. The exact path's SonicMoE call fuses gather, GEMM and
SwiGLU, so the pre-activation never reaches memory. The runtime therefore
replays `moe_general_routing_inputs` with the same routing metadata, QuACK
kernels and top-k combine, and hands both grouped GEMMs the shift as their
addend `C` (`D = A·B + C`, added to the float32 accumulator before the
epilogue). A pair without expert factors (no accent, or a `/1` adapter) has a
zero row in `C`, and the kernels with a zero addend return the plain call's
bits.

The shift itself is a Triton kernel (`zonos3_sgl/kernel/triton/expert_lora.py`).
Each program takes one pair and one output tile, reads that pair's
`(adapter, expert)` factors, and computes `α · U (D x)` in float32 in a fixed
order. A pair's bits therefore do not depend on the rest of the batch, and the
kernel has static shapes for the CUDA graphs. Adapters are grouped by expert
rank: each rank keeps one `[adapters·E, in, r]` / `[adapters·E, r, out]` stack
per weight and costs one launch per weight. Pairs off a rank load nothing.
The engine probes these kernels at load like the other side paths, with 1-256
pairs against 1024 on every slot.

Exact mode stays batch-invariant with adapters loaded. A request without an
accent has an all-zero mask row, so its shift is an exact zero. For accented
requests the engine probes every adapted op at load with 1-256 rows against a
1024-row product. Any op whose cuBLAS kernel depends on the row count runs its
side path as fixed 16-row GEMMs instead (on H100, the LM head).

`servers/sgl/scripts/accent_parity.py` (wrapped by `servers/sgl/tests/test_accent_adapters_gpu.py`)
checks this offline on seeded requests and writes a `report.json`:

- (a) requests without an accent, batched with accented requests, against the
  engine without adapters (bit-equal logits and codes);
- (b) every request of a batch mixing accents and strengths against the same
  request run alone (bit-equal);
- (c) an adapter at strength 1 against its merged checkpoint
  (the training code's merged export, then `tools/convert_lingua_checkpoint.py`),
  teacher-forced on the adapter's codes. These agree closely but not bit for
  bit: the served sum and the merged bf16 weights round differently, and the
  MoE top-k turns some of those ulps into different experts. The base model on
  the same streams gives the scale of the adapter's whole effect.

Requests cycle over no speaker prompt, the raw reference prefix
(`--reference-audio`) and the speaker-embedding row (`--speaker-embedding`),
with unknown condition rows; `--merged-model` is optional. The test wrapper
reads `ZONOS3_TEST_MODEL`, `ZONOS3_TEST_ACCENT_DIR`, `ZONOS3_TEST_MERGED_MODEL`,
`ZONOS3_TEST_MERGED_ACCENT`, and optionally `ZONOS3_TEST_MIXED_ACCENTS`,
`ZONOS3_TEST_REFERENCE_AUDIO`, `ZONOS3_TEST_SPEAKER_EMBEDDING` and
`ZONOS3_TEST_PARITY_OUT`; run it in the exact-mode environment of `servers/sgl/scripts/env.sh`.

Speaker and condition conditioning are prompt-side. The speaker-embedding row
(`speaker_projection`, Linear 192 → 1024), the three condition rows
(`condition_embeddings`) and the raw reference prefix replace input rows
before the transformer, so the adapters need no path for them. Those tensors
are not LoRA targets: a factor on `speaker_projection` is refused as outside
`TARGETS`, and an exact shift of it or of a condition table is refused as
neither a linear factor nor a router bias (`servers/sgl/tests/test_accent_adapters.py`).

A merged checkpoint (release LM plus an adapter, merged by the training code)
converted with `tools/convert_lingua_checkpoint.py`
the way the release LM was converted (its `conversion.json` holds the converter
arguments) is the release model plus the adapter, which is what parity check (c)
compares against.

## Steering

Request semantics are in [guide: Steering controls](guide.md#steering-controls).

### Hidden-space steering in the engine

For a hidden-space set (`manifest.json`:
`"space": "attnres_source"`, `"source_index": k`), on every generated frame the
request's delta is added to source `k` of the Full AttnRes bank (0 = embedding,
`2i+1` = attention/GDN output of layer `i`, `2i+2` = FFN/MoE output of layer `i`)
before later sublayers and the output head route from it. Reference and text rows
are never steered. The add is captured in the decode CUDA graphs with a static row
buffer; with no control requested the buffer is zero and the logits are
bit-identical to a server without a bundle. Sets may steer different sources: the
engine keeps one row-buffer slot per steered source and the request delta carries
one hidden-size slot per source. `raw` sets move the speaker embedding instead
(their deltas summed, the result rescaled to the embedding's norm) and need
`speaker_mode: embedding`. A set is `raw` or `attnres_source`; any other space is
refused at load.

### How the shipped direction sets were fitted

Directions are measured, not trained: clips are teacher-forced through the offline
engine, every bank source is mean-pooled over the clip's own frames, and a direction
is a contrast of those means (e.g. style − neutral) at one source. The shipped sets
(`<set>_mect60k_x10/`) are refits of the original mect60k sets with the same specs on
about ten times as much natural speech (spontaneous corpora included; each set's
`manifest.json` names its corpora, `report.json` its fit), every direction rescaled to
the norm of the original fit so doses and the studio's ranges carry over; the notes
below name the original fits' data. The shipped
`assets/steering/emotion_directions_mect60k_axes_x10/` (bundle set `mood`) uses the GDN output
of layer 9 (`source_index` 19, `L9.attn`) and folds the style directions into
orthogonal conceptual axes: valence, arousal, dominance, loudness, tempo, clarity
and articulation. `articulation` is the component enunciated and confused share,
which clarity (their difference) cancels: positive is slow, deliberate delivery,
negative quicker, looser speech.

`assets/steering/speaker_accents_mect60k_x10/` (raw, bundle set `accents_l1`): the direction is the
mean over native L1 speakers minus the mean over native
English speakers of the embedding (clips averaged per recording first; gender and
age, `assets/steering/speaker_attributes_mect60k_x10/`, projected out). The embedding has no
text in it, so unlike the hidden native-minus-English direction it does not carry
a language switch.

`assets/steering/mandarin_mect60k_x10/` (raw, `cmn`) and
`assets/steering/mandarin_hidden_mect60k_x10/` (L7.attn, `cmn`) are the speaker-embedding direction and the
hidden native − English direction for Mandarin. Neither
gives Mandarin, Korean or Japanese alone. Served together the two add up for
Mandarin, at the price of the voice.

`assets/steering/english_accents_mect60k_x10/` (raw, bundle set `accents_en`) holds british, american, scottish, irish,
australian, indian and welsh, each relative to American English (american is
the reverse of british, for British voices). They are fitted from speakers with a
known accent, each contrast inside one corpus so its recording conditions cancel:
VCTK (accent column), VoxCeleb1 dev (nationality) and OpenSLR 83 English dialects
(contrasts against southern English, chained through british), gender and age
projected out.

`assets/steering/voice_quality_mect60k_x10/` (raw, bundle set `texture`) holds deep,
shrill, husky, raspy, soft and authoritative: ParaSpeechCaps' human speaker tags
(VoxCeleb / EARS / Expresso speakers on their original recordings), tagged minus
untagged speakers inside each corpus and gender, gender and age projected out. The
tags are noisy and the directions short.

`assets/steering/pitch_mect60k_x10/` (raw axes) holds pitch and pitch_range, fitted on
LibriTTS-R speakers with their measured F0 (pYIN): the least-squares slope of the
embedding on the speaker's log F0 (pitch) or F0 spread in semitones (pitch_range)
inside each gender, times the P10-P90 span (dose 1 = from a low-pitched to a
high-pitched speaker of the same gender), gender and age projected out.

`assets/steering/channel_mect60k_x10/` (raw axis `clean`) is the within-speaker change of the
embedding with ParaSpeechCaps' recording-cleanliness tag (very noisy … very clean,
per clip): every VoxCeleb clip minus its speaker's mean clip, regressed on the tag
minus the speaker's mean tag (`"group_by": "clip", "center": "speaker"`), so the
direction is the channel, not who records in it.

`assets/steering/ears_styles_mect60k_x10/` (attnres_source, L9.attn like the `mood` axes)
holds whisper, loud, tempo (fast − slow), pitch (high − low) and conversational
(spontaneous − read): EARS's speakers read the same passage and sentences in every
style and talk freely, so each direction is a within-speaker contrast
(`"center": "speaker"`) with no text or channel difference.

`assets/steering/expressive_mect60k_x10/` (attnres_source L9.attn, axis `expressive`) is the
within-speaker change of the bank with a clip's F0 spread: LibriTTS-R readers'
livelier and flatter clips. Livelier
readings are also higher and slower in LibriTTS-R, and the direction keeps that.

`assets/steering/disfluency_mect60k_x10/` (attnres_source L9.attn, axis `disfluent`):
DisfluencySpeech's speaker reads Switchboard turns with their fillers, repetitions
and restarts; each clip is teacher-forced with the FLUENT transcript, so the bank
sees hesitations the text does not ask for, and the direction is the regression on
how many.

`assets/steering/emphasis_nv45_nodur/` (attnres_source L9.attn, bundle set `emphasis_nv_nodur`, direction
`emphasis`; fitted on the NV + phoneme edit checkpoint) is the mean over 636 Expresso pairs of an emphasized word
minus the same word in the speaker's plain take, with the log-duration slope projected out.

EARS's speakers also read the same sentences in 22 emotions and neutral.
Within-speaker emotion − neutral gives angry, sad, happy (amusement), fearful,
disgusted and surprised (amazement) both in the speaker embedding
(`assets/steering/speaker_emotions_mect60k/`, raw, gender and age projected out) and at
L9.attn (`assets/steering/ears_emotions_mect60k/`, attnres_source).

`assets/steering/ears_moods_mect60k_x10/` holds every EARS emotion against neutral in the same
space and spec style as `ears_emotions_mect60k` (attnres_source L9.attn,
`"group_by": "clip"`, `"center": "speaker"`; its six directions are identical):
adoring, confused, content, cute, desirous, disappointed, distressed, embarrassed,
ecstatic, guilty, interested, pained, proud, realizing, relieved and serene next to
angry, sad, happy, fearful, disgusted and surprised, plus `scared`, fear against the
other 21 emotions. The directions share one large "emotional reading vs neutral"
component, and most of them add that component and arousal without their specific
target.

`assets/steering/accent_directions_mect60k/` holds Hindi only, at L7.attn
(`source_index` 15): accented English minus English, from English-labelled clips
that MMS-LID hears as most Hindi-like against native English clips. The other
accents did not work as one hidden vector (native speech minus English mostly
switches language before the accent moves).

### Steer then hold: schedule and measurements

At generated frame f (codec frames, 15 per second; a LACI continuation counts its history, an edit fill its
own first frame) the weight is 1 before the onset, `hold` from onset + ramp on, `1 − (1 − hold)·(f − onset)/ramp` in
between (ramp 0 is a step); both CFG branches of a steered twin get the same weight. The `exempt` directions (pace:
span placement and edit lengths assume it), `control_spans`, raw (speaker-embedding) directions and server-side
deltas (a LACI take's tempo bucket) stay constant.

The shipped defaults (onset 4 s, ramp 1 s, hold 0.35; `delivery.tempo` and `delivery.whisper` exempt) were measured
on the release LM with the x10 bundle. On three 25–32 s passages (Thaddeus and Rupert, 4 seeds each) a stacked voice
design (raw identity / pitch / texture / room controls plus `delivery.conversational` 3, `expressive.expressive` 4,
`mood.valence` 2, `mood.arousal` 2) held constant climbed in 19 of 24 takes (a 4 s window ≥ 6 semitones above the
first 8 s; neutral 0/24), +7.2 st and +7.0 dB over the neutral take after the first 5 s. Holds 0.25 / 0.35 / 0.5
climbed in 0 / 0 / 2 takes (+0.2 / +0.6 / +1.9 st, +4.9 / +5.2 / +5.5 dB), with WER and identity unchanged; the
default is the largest hold with neutral's climb count. A hidden direction stays constant when none of its
single-direction tests raises loudness or median F0 and it loses its effect once released (32 short lines, 4 voices):
the whisper kept only 43% of its voicing drop after the onset without raising level or pitch; `emotions.sad` raises
median F0 on the x10 sets and the Mandarin accent's hidden part keeps its effect, so both release.

### Span steering by reading position

The tokenizer maps each span's characters through text normalization
to the text tokens that spell them (SuperBPE offsets, every id of a pronunciation group, UTF-8 bytes on byte
checkpoints); while the request decodes, the LACI alignment heads of its language (`--laci-heads`; the probe of
`zonos3_sgl.tts.align_probe`, inside the decode CUDA graphs, no LACI retries unless `laci=true`) give one reading position
per step, smoothed (median of 3, monotone, at most +1 token per step). A span ramps in (3 frames) once the smoothed
position reaches its first token − `control_span_lead` (default 1) and out once it reaches the token after it +
`control_span_lag` (default 0). The readout reaches the host with the step's tokens, so a frame is decided on readouts
one or two steps old; every decision is recorded per frame, so the speaker-CFG twin steers exactly like its request and
a LACI rollback replays from its row.

### Extended speaking-rate buckets: derivation

The pace control (`delivery.tempo`, hidden L9.attn) can stand in for the speaking-rate condition row: every bucket of
the row has a tempo dose that reproduces it (agreeing across English and French at the same WER and identity), the
dose is continuous and reaches further than the row in both directions, and the rate row is left unknown while
steering (a set bucket damps the steering).

The bundle's `rate_extension` block was measured on the original `ears_styles_mect60k` tempo direction and on a
ten-bucket speaking-rate row, whose buckets 0-9 the bullets below refer to.

- **Nominal edges** continue the row's at its mean geometric step (×1.097): bucket 0 becomes [9.22, 10.12) B/s,
  −1 [8.41, 9.22), … and 9 becomes [21.21, 23.26), 10 [23.26, 25.52), …; a language's lowest / highest bucket stays
  open-ended.
- **Doses**: bucket −k / 9+k targets the achieved rate (vs the unsteered render) of the row's own achieved ladder
  continued k steps; the dose is interpolated on the measured dose curve of the language.
- **Kept buckets** are those whose achieved rate lands within one step of the target at an unchanged WER, collapse rate
  and identity, contiguous from the row. The per-language table is in
  [guide: Extended speaking-rate buckets](guide.md#extended-speaking-rate-buckets-rate_extension).

## Tensor Parallelism

Weights are sharded across TP ranks:

| Component | Sharding |
|-----------|----------|
| `wq`, `gater` | Column-parallel (split output dim) |
| `wkv` | Split KV dim (dim 1 of 3D weight) |
| `wo` | Row-parallel with all-reduce |
| `w_in` (FFN) | Split intermediate dim (dim 1) |
| `w_out` (FFN) | Row-parallel with all-reduce |
| MoE `gate_up_proj` | Split intermediate (dim 1) |
| MoE `down_proj` | Split intermediate (dim 2) |
| Embeddings | Vocabulary-parallel |
| `multi_output` | Output vocabulary sharded |
| `attention.temp` | Split heads (dim 1) |

Sharding logic in `zonos3_sgl/models/weight.py`. Communication via NCCL all-reduce in output projections.

## Configuration

Key `ServerArgs` fields (in `zonos3_sgl/server/args.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `tts_n_codebooks` | `9` | Number of audio codebooks (replaced by the checkpoint's `n_codebooks`, 10 for ZONOS3 Mini, while left at the default) |
| `tts_audio_pad_id` | `1025` | Audio padding token (replaced by the checkpoint's `audio_pad_id`, 4097, while left at the default) |
| `tts_text_vocab` | `None` | Text vocabulary size (auto-detected from config) |
| `tts_sample_rate` | `44100` | Output audio sample rate |
| `tts_audio_codec` | `opus_dual_3n_600k_sched600k_r4096_step600000` | Matched codec, taken from the checkpoint's `audio_codec` |
| `tts_default_voices_dir` | `None` | Optional folder of audio or `.npy`/`.npz` embedding files listed as default voices in `GET /v2/speakers` and the studio's speaker picker (`servers/sgl/scripts/serve.sh`: `default_voices/`) |
| `studio_dir` | `<repo>/studio/dist` | Studio build served at `/` |
| `steer_bundle` | `None` | Steering bundle JSON behind `controls` / `control_spans` and `GET /v2/controls` (`servers/sgl/scripts/serve.sh`: `steering/studio_bundle_mect60k_x10.json`) |

Model-level config is loaded from checkpoint via `cached_load_checkpoint_config()` (`zonos3/utils/hf.py`) and `ModelConfig.from_checkpoint_config()` in `zonos3/models/config.py`. Key fields: `n_codebooks`, `codebook_size`, `text_vocab`, `eoa_id`, `audio_pad_id`, `loss_softcap`, `moe_n_experts`, `moe_start_from_layer`, `moe_router_dim`.

## Key Files

| Path | Role |
|------|------|
| `zonos3_sgl/server/api_server.py` | HTTP endpoints, `FrontendManager` |
| `zonos3/server/schemas.py` | Request models (the v2 `GenerateRequest`, `SpeechRequest`, …), shared by the Python servers |
| `zonos3/tts/plan.py` | v2 request rules and prompt plan of every Python server (the frontend's checks, the tokenizer worker's row helpers) |
| `zonos3/generation.py` | Generation registry (`registries/generation.json`): sampling defaults and request limits |
| `zonos3_sgl/server/launch.py` | Process spawning |
| `zonos3_sgl/server/args.py` | `ServerArgs` configuration |
| `zonos3_sgl/tokenizer/server.py` | Tokenizer/detokenizer worker loop |
| `zonos3/tts/prompt.py` | Prompt rows (speaker, condition, reference prefix, text), shear, silence prefill rows |
| `zonos3/tts/superbpe.py` | SuperBPE text convention, phoneme groups |
| `zonos3_sgl/tokenizer/vocoder.py` | `TTSVocoderManager`: undelay, whole / prefix decode |
| `zonos3/tokenizer/matched_codec.py` | Codec registry, `ISTFTCodec` decoder load and decode, per decoder id |
| `zonos3/tokenizer/dual_reference.py` | Reference-audio encoder (`MossAutoencoder`) |
| `zonos3/codec/` | Vendored dual codec: encoder adapter, quantizer, causal MR iSTFT decoder |
| `zonos3_sgl/engine/engine.py` | `Engine.forward_batch_tts()` |
| `zonos3_sgl/models/zonos3.py` | `Zonos3ForCausalLM` |
| `zonos3/models/config.py` | `ModelConfig` |
| `zonos3_sgl/models/weight.py` | Weight loading and TP sharding |
| `zonos3_sgl/tts/sequence.py` | `TTSSequence`, EOS detection |
| `zonos3/tts/sampler.py` | `sample_tts()` |
| `zonos3_sgl/tts/llm.py` | `TTSLLM` offline generation interface |
| `zonos3_sgl/message/tts.py` | Message dataclasses, `TTSSamplingParams` |
| `zonos3_sgl/attention/` | FlashInfer / FlashAttention backends |
| `zonos3_sgl/layers/` | Linear, norm, RoPE, MoE layers |
| `zonos3/reference/` | Plain-PyTorch reference LM and synthesis (`python -m zonos3.reference`) |
