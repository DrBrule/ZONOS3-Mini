# ZONOS3 feature guide

The sections below give the request fields, server flags and behavior of each ZONOS3 Mini feature beyond plain
speech synthesis. The endpoints are in [API reference](api.md), LACI and short texts are in
[Features](features.md), and [TTS System Architecture](tts_architecture.md) explains how the servers implement
these features.

## Contents

- [Voice cloning](#voice-cloning)
- [Speech editing](#speech-editing)
  - [Setup](#setup)
  - [Endpoints](#endpoints)
  - [Edit request fields](#edit-request-fields)
  - [Long clips](#long-clips)
  - [Steer an edit](#steer-an-edit)
  - [Languages](#languages)
  - [Workers](#workers)
  - [Python API](#python-api)
- [Accent adapters](#accent-adapters)
- [Pronunciation input (phoneme tokens)](#pronunciation-input-phoneme-tokens)
  - [Load the text rows](#load-the-text-rows)
  - [Request text](#request-text)
  - [Groups in speech edits](#groups-in-speech-edits)
  - [In the studio](#in-the-studio)
  - [Phonemize](#phonemize)
- [Start and pause tokens](#start-and-pause-tokens)
- [Literal angle-bracket text](#literal-angle-bracket-text)
- [Steering controls](#steering-controls)
  - [Steer then hold](#steer-then-hold)
  - [Direction kinds](#direction-kinds)
  - [Set calibration](#set-calibration)
  - [Shipped direction sets](#shipped-direction-sets)
  - [Sets outside the steering bundle](#sets-outside-the-steering-bundle)
- [Zonos Voice Studio](#zonos-voice-studio)
  - [Span steering by reading position](#span-steering-by-reading-position)
  - [Projects on the server](#projects-on-the-server)
- [Plugins](#plugins)
  - [Extended speaking-rate buckets (`rate_extension`)](#extended-speaking-rate-buckets-rate_extension)

## Voice cloning

A request clones a voice in one of two modes. In embedding mode the speaker encoder turns the reference audio
into one 192-d speaker embedding, which fills one prompt row. In reference mode the reference audio is encoded
to reference codes that precede the text in the prompt. `speaker_mode` selects the mode; its default is the
checkpoint's, which is `reference` on the release LM.

A request supplies the reference as audio in `speaker_audio_base64` or as reference codes in
`speaker_reference_codes`, which are `[frames, 10]` codec ids. Reference codes keep their first 900 frames
(60 s), and a default voice's `<stem>.codes.npy` is cut the same way.

With `decode_context`, the decoder reads the last `decode_context_frames` reference frames as left context;
`decode_context_frames` is an integer with default 30, and 0 also means 30. A `speaker_cfg_scale` above 1 mixes
the prompt with and without the reference. A request produces the same codes alone and inside a batch.

## Speech editing

Speech editing regenerates the changed words of a recording, and the rest of the clip keeps its exact codec
frames. It runs on the release LM with the speech-edit adapter. Only edit fills use the adapter, so every other
request produces the same audio as without it.

### Setup

1. Load the speech-edit adapter. On the CUDA server, download it and pass its directory:

   ```bash
   hf download Zyphra/ZONOS3-Mini lm/zonos3_mini/edit_lora/z3m_omni3b_edit_lora.pt --local-dir .
   mkdir -p edit-adapter && cp lm/zonos3_mini/edit_lora/z3m_omni3b_edit_lora.pt edit-adapter/edit.pt
   servers/sgl/scripts/serve.sh --accent-adapters edit-adapter
   ```

   The ggml and MLX servers load `gguf/backbone-ptq-edit.gguf` with `gguf/edit/accents.gguf`; their READMEs
   give the commands: [zonos3.cpp](../servers/ggml/README.md#speech-editing) and
   [zonos3-mlx](../servers/mlx/README.md#speech-editing). A checkpoint fine-tuned for editing is served
   directly with `SERVE_MODEL=<checkpoint>`.

2. Install FuseAlign, which supplies the word timings. `uv sync` installs `fusealign-hub` from its submodule.
   The checkpoint is in the public Hugging Face repo `Zyphra/fusealign` and must be in the local cache, because
   the workers run with `HF_HUB_OFFLINE=1` unless that variable is set:

   ```bash
   git submodule update --init third_party/fusealign-multilingual
   uv sync
   hf download Zyphra/fusealign fusealign-multilingual-cjkft-step2500.safetensors
   ```

3. Optional: install speech recognition. `bash tools/setup_asr_env.sh` creates `.venv-asr` with Qwen3-ASR
   (`Qwen/Qwen3-ASR-1.7B`) and downloads its weights. With it, the server transcribes clips sent without their
   transcript and runs `verify`.

`/v2/capabilities` reports the result.

|Capability|Meaning|
|---|---|
|`speech_editing`|`true` when the LM has the edit rows, `fusealign_hub` imports and the reference encoder is available|
|`edit_transcribe`|`true` when speech editing is available and `.venv-asr` exists|
|`edit_max_spans`|regions per request: 3|
|`edit_frame_rate`|codec frames per second: 15|
|`edit_max_seconds`, `edit_align_max_seconds`|longest clip that the server edits, and that it transcribes and aligns: 150 s|

When `speech_editing` is `false`, the edit endpoints answer 400. A `<voice>.txt` next to a default voice becomes
that voice's `transcript` in `GET /v2/speakers`.

Without speech recognition, three kinds of request answer 503 before any encode or generation. The detail
starts with `Speech recognition is not available on this server: ` and names the fix.

|Request|Fix named in the 503 detail|
|---|---|
|`POST /v2/edit/transcribe`|`send the transcript to /v2/edit/align instead`|
|a request with `verify: true`|`set verify to false`|
|a `/v2/edit` that reads the clip's words and has neither `original_text` nor `original_words`|`send the clip's transcript as original_text`|

A `/v2/edit` request reads the clip's words when it has no `spans`, or when the clip is longer than
`max_context_seconds`.

### Endpoints

The three endpoints take the clip as base64 audio.

|Endpoint|Body|Returns|
|---|---|---|
|`POST /v2/edit/transcribe`|`{audio_base64}`|the Qwen3-ASR transcript, `duration` and `words`; needs `edit_transcribe`|
|`POST /v2/edit/align`|`{audio_base64, text}`|the same fields for the given text|
|`POST /v2/edit`|`{audio_base64, target_text}` plus the fields of [Edit request fields](#edit-request-fields)|the edited takes|

A word object is `{word, start, end, confidence, tokens}`, where `tokens` are the word's edit tokens. Words are
the whitespace words of the text. In Japanese and Chinese, a word is one ICU word of a Han or kana run, or one
Latin or digit run.

`confidence` is FuseAlign's presence probability, and a word that FuseAlign did not place keeps its fallback
interval. An overlap between neighbors is split at its midpoint, so the words can be sent back as
`original_words`. Punctuation-only words are left out.

Transcripts are compared as edit tokens: upper-cased, with every letter or digit of any script and `'` kept, and
everything else a word break.

### Edit request fields

`POST /v2/edit` takes these fields besides `audio_base64` and `target_text`.

|Field|Default|Meaning|
|---|---|---|
|`spans`|none|regions as `[start, end]` seconds in the original, ascending, at most `edit_max_spans`; without them the regions come from a word diff of `original_text` against `target_text`|
|`original_text`|the Qwen3-ASR transcript|the clip's transcript; omitting it needs `edit_transcribe`|
|`original_words`|the server's alignment|checked word timings of `original_text`|
|`margin_seconds`|0.08|padding on each side of a region from the word diff|
|`max_span_seconds`|30|longest fill of a region|
|`min_fill_ratio`|0.7|shortest fill, as a share of the region's expected length; 0 turns it off|
|`max_context_seconds`|16|longest clip that is edited whole; at most 20|
|`context_seconds`|4|unedited audio on each side of a region in a window|
|`num_takes`|1|takes to return, at most 4; take `k` uses `seed + k`|
|`verify`|`false`|report Qwen3-ASR WER and CER and FuseAlign words per take; needs `edit_transcribe`|
|`speaker_embedding`|on|the clip's speaker embedding fills the speaker row; `false` leaves the row out|
|`controls`|none|control ids and doses for every region|
|`region_controls`|none|one `{id: dose}` or `null` per explicit span|
|`language`|`en_us`|word tokenization and recognition language|
|`text_normalization`|`true`|normalize `target_text`|
|sampling fields, `condition_buckets`, `condition_values`, `decoder`, `steer_*`|as on `/v2/generate`|as on `/v2/generate`|

The response has these fields.

|Field|Meaning|
|---|---|
|`spans`|the edited regions, in seconds of the original|
|`original_text`|the transcript that the edit used|
|`words`|the word objects of the original|
|`contexts`|the windows, in seconds of the full clip; `null` when the clip was edited whole|
|`expected_seconds`|expected fill length per region|
|`takes[].audio_base64`|the edited clip as a 16-bit 44.1 kHz WAV|
|`takes[].edited_regions`|where each region lies in the result|
|`takes[].fill_seconds`|length of each fill|
|`takes[].stopped`|per region, whether the fill ended on the span-end token; `false` means it ran out of budget or produced an invalid frame|

`min_fill_ratio` masks the span-end token until a fill reaches that share of its region's `expected_seconds`.
`expected_seconds` is the region's length scaled by the ratio of target to original text length, counted in edit
token characters of the region's words. The server reads those words from its FuseAlign alignment of
`original_text`; without `original_text`, `expected_seconds` is the region's own length.

`original_words` is a list of `{word, start, end, tokens?, confidence?}`: the word objects of the align and
transcribe endpoints, in order, with timings that the caller has checked or corrected. They replace the server's
alignment of `original_text` for windows, expected fill lengths and the word diff, and come back as `words`.

The words must spell `original_text` as edit tokens; for a word without `tokens`, the server derives them from
the word. Times must be finite, ascending, non-overlapping and inside the clip, with `start < end`; otherwise
the request is a 400.

### Long clips

A clip longer than `max_context_seconds` is edited on windows, because the speech-edit adapter was trained on
2-20 s clips. A window is a region plus `context_seconds` of unedited audio on each side, cut halfway between
aligned words of the original. An edge that falls inside a changed block of the word diff moves out to the
block's edge, and overlapping windows merge.

A window's fill prompts hold only its codes and its stretch of the target text. Every fill splices into the full
clip, which is decoded once with the full clip's speaker row and per-region steering. `contexts` lists the
windows.

Clips over 30 s, and texts over 1022 UTF-8 bytes, are aligned in speech chunks of at most 25 s. When speech
recognition is installed, Qwen3-ASR transcribes each chunk in the request's language and the timed transcripts
anchor the alignment. Without speech recognition, FuseAlign routes the text by its own speech map, which is
less accurate on long clips. FuseAlign's built-in Whisper anchoring is never used.

A clip over 30 s that the server transcribes itself is transcribed chunk by chunk in the same way. Chunk
transcripts are joined with a space, except in Japanese, Chinese and Cantonese.

### Steer an edit

`controls` takes the control ids of `/v2/generate` and applies to every region. `region_controls` adds to
`controls` for one region, and doses are summed per id. Each region is then resolved like `controls` of
`/v2/generate`:

- A raw direction moves the clip's speaker embedding in the speaker row; without a speaker row it is a 400.
- A hidden direction acts on the frames that the fill generates.

The prompt before the fill is not steered, so the context keeps its audio and only the regenerated words take
the delivery. Unknown ids, `region_controls` without `spans` and `region_controls` of the wrong length are 400s.

### Languages

`language` on `/v2/edit`, `/v2/edit/transcribe` and `/v2/edit/align` is a server language code. It selects
FuseAlign's word tokenization and the Qwen3-ASR language of every recognition in the request. Editing and
alignment accept every server language. Recognition works for the languages that the ASR worker supports, and
any other language is a 400 when the request needs recognition.

### Workers

Each server runs one Qwen3-ASR worker and one FuseAlign worker, and plugins share them. A worker starts on its
first call and restarts once when it exits. An unavailable worker is a 503, and input that a worker refuses is a
400. No request contacts the ASR worker unless it recognizes speech.

These variables set the workers' interpreter and devices.

|Variable|Default|
|---|---|
|`ZONOS3_ASR_PYTHON`|`.venv-asr/bin/python`|
|`ZONOS3_ASR_DEVICE`|`mps` on macOS; else `cuda:0` when available, else the CPU|
|`ZONOS3_FUSEALIGN_DEVICE`|`cuda:0` when available, else the CPU|

A request with explicit `spans`, a clip of at most `max_context_seconds` and no `verify` needs no speech
recognition. It needs no alignment either when it has no `original_text` or sets `min_fill_ratio` to 0.

### Python API

`TTSLLM.edit(codes, target_text, spans)` edits a clip's codec codes offline. `spans` are
`(start_frame, length_frames)` pairs, ascending, at most `edit_max_spans`.
`servers/sgl/scripts/edit_speech.py` wraps the call: it takes `--span START END` in seconds or derives the spans
from a word diff, and it has `--max-context-seconds` and `--context-seconds`.

## Accent adapters

An accent adapter is a low-rank adapter that a server applies on top of the LM for the requests that select it.
On the CUDA server, `--accent-adapters DIR` loads the adapters of a directory:

```bash
servers/sgl/scripts/serve.sh --accent-adapters <adapter dir>
```

`DIR` holds `<accent>.pt` files in the `zonos-accent-lora` format, which
`model/src/zonos3/models/accent_format.py` defines. The file stem is the accent's name, and `/v2/capabilities`
lists the loaded names in `accents`. A request selects an accent with `accent` and scales it with
`accent_strength`; an unknown name is a 400.

A request runs base + strength × adapter. Requests with different accents or strengths, and requests with no
accent, share batches. An accent applies in both speaker modes and with speaker CFG. Tensor parallelism is not
supported with adapters.

The format has three versions.

|Format|Adds|Served|
|---|---|---|
|`zonos-accent-lora/1`|linear factors|everywhere|
|`zonos-accent-lora/2`|per-expert MoE factors|only with `ZONOS3_USE_TRAINING_SONIC_EXPERTS=1`, which `servers/sgl/scripts/env.sh` sets on Hopper GPUs; refused elsewhere|
|`zonos-accent-lora/3`|the vocabulary rows of the speech-edit adapter|at most one per directory; not listed in `accents`, and no request can select it|

The ggml and MLX servers read `accents.gguf`, which `tools/convert_gguf.py accents <adapter dir>` writes. The
public release ships no accent adapter other than the speech-edit adapter.

Accent directions are steering controls, not accent adapters; [Shipped direction sets](#shipped-direction-sets)
lists them. [Accent adapters](tts_architecture.md#accent-adapters) in the architecture document explains how
the servers apply an adapter.

## Pronunciation input (phoneme tokens)

A word can be written as its IPA inside a pronunciation group. The release text rows,
`hf://Zyphra/ZONOS3-Mini/lm/zonos3_mini/text_rows/pause_tokens_s5000.pt`, hold 143 rows: 138 phoneme rows
(open, close and 136 IPA symbols), a start row and four pause rows.

Only those rows were trained, so text without groups or markers renders exactly as on the release LM. In
evaluation, a name written as IPA was said as written 95 % of the time, against 18 % when the release LM read
the same IPA as text. The word error rate on plain text stayed the same.

### Load the text rows

Each runtime loads the release text rows in its own way.

|Runtime|Loads the rows|
|---|---|
|CUDA server, `servers/sgl/scripts/serve.sh`|when `SERVE_MODEL` is unset; `SERVE_TEXT_ROWS=<path or hf://>` names the rows for another checkpoint, and `SERVE_TEXT_ROWS=` loads none|
|`python -m zonos3.reference`|with the release LM; `ZONOS3_RELEASE_TEXT_ROWS` overrides the location, and `--phoneme-rows ''` loads none|
|the Space|always|
|ggml and MLX servers|from the backbone file, which is converted with `--phoneme-rows`; `gguf/backbone-ptq-edit.gguf` has them|

```bash
# CUDA server: release LM, speech-edit adapter, release text rows, lexicon.
PHONEME_LEXICON=<lexicon dir> servers/sgl/scripts/serve.sh --accent-adapters <dir with the edit adapter>
# Another checkpoint: name the rows as a path or an hf:// location.
SERVE_MODEL=<LM> SERVE_TEXT_ROWS=<pause_tokens_s5000.pt> servers/sgl/scripts/serve.sh
# Reference implementation.
uv run python -m zonos3.reference --model <LM> --phoneme-rows <pause_tokens_s5000.pt> --text '...' -o out.wav
# ggml and MLX servers: the rows go into the backbone file.
uv run --package zonos3-tools python tools/convert_gguf.py lm <LM dir> --outtype q4_0-exp-q8_0-mix \
    --edit-adapter <z3m_omni3b_edit_lora.pt> --phoneme-rows <pause_tokens_s5000.pt> -o backbone-ptq-edit.gguf
# A converted checkpoint that carries the rows on disk.
uv run --package zonos3-tools python tools/add_text_extra_rows.py \
    --base <checkpoint> --rows <pause_tokens.pt> --out <checkpoint>
```

[Extra text rows](tts_architecture.md#extra-text-rows) describes where the rows sit in the text vocabulary.

### Request text

In `text`, a group is U+E000, the IPA, then U+E001. It encodes as the open id, one id per IPA code point and the
close id, and the text between groups is encoded as usual. Text normalization leaves groups as typed and
verbalizes the text between them.

A malformed group, a symbol outside the inventory and a group sent to an LM without phoneme tokens are each
refused with a 400. The rule covers `text` of `/v2/generate`, `input` of `/v1/audio/speech` and `target_text`
of `/v2/edit`.

### Groups in speech edits

`target_text` of `/v2/edit` takes groups, while `original_text` and `original_words` stay plain. With `verify`,
a group matches any one recognized word, and `verify.groups` lists `{ipa, heard}` for every group in order. In
expected fill lengths, a group counts 1.4 letters per IPA symbol.
[Speech editing pipeline](tts_architecture.md#speech-editing-pipeline) gives the reasons.

### In the studio

In Zonos Voice Studio (the studio), type a pronunciation in X-SAMPA inside double square brackets, as in
`I met [[S@vQn]]`. The studio shows the IPA that each group becomes, flags symbols that the LM has no token for,
and sends U+E000 IPA U+E001 groups. Without phoneme tokens, `[[...]]` is sent as typed.

Stress marks, syllable dots and spaces are dropped, as in training. A lone `_` is a tie, so `t_S` is t͡ʃ.
Symbols without an X-SAMPA spelling, such as superscript tone digits, `ᵝ` and `ˣ`, are typed as they are.

### Phonemize

[`POST /v2/phonemize`](api.md#post-v2phonemize) returns each word's pronunciation and whether it was guessed.
The studio's **Pronounce** button uses it to replace the selected words with their lexicon pronunciation. The
endpoint needs `--tts-phoneme-lexicon`, which `PHONEME_LEXICON` sets in the serve scripts. The lexicon
`lexicons/gold_phonemes_v1` is in no model repo; the Space ships it.

Only words whose deciding source has a single pronunciation come from the lexicon. Hiragana, okurigana and
single kanji or hanzi stay as text, as in training. In the space-delimited languages, espeak-ng guesses
ambiguous and unknown words, and the studio lists them as guesses to check. Chinese, Japanese, Korean and
Vietnamese are never guessed.

espeak-ng is GPL-3.0, so it is in the opt-in `phonemize` dependency group, installed with
`uv sync --locked --group phonemize`. The serve scripts run with that group, and without espeak-ng unknown words
stay as text.

`/v2/capabilities` lists the phone symbols, `phonemize_languages` and `phonemize_guess_languages`. The two
language lists depend on the loaded lexicon and the installed espeak-ng voices.

## Start and pause tokens

The release text rows end with a start row and four pause rows. `/v2/capabilities` reports them as
`start_token` and `pause_tokens`.

The start marker U+E002, once and as the first character of the text, asks for speech from the first generated
frame. `start_immediately: true` puts the marker in front of the normalized text unless it is already there. The
marker was trained without the silence prefill, so with it `silence_prefill_frames` defaults to 0; an explicit
`silence_prefill_frames` applies.

A pause marker goes after a word and its attached punctuation, as in `slid,\ue004 on`, and asks for a pause of
about its nominal length. A marker past the LM's pause rows is a 400.

|Marker|Nominal pause|
|---|---|
|U+E003|0.2475 s|
|U+E004|0.495 s|
|U+E005|0.9899 s|
|U+E006|1.9799 s|

Text normalization keeps both kinds of marker in place and verbalizes the text around them. `target_text` of a
speech edit refuses them.

In the studio, the "Start immediately" toggle on a Direct line sends `start_immediately`; the toggle is shown
when the LM has a start token. A pause is typed as seconds in angle brackets, as in `<0.5>`, and the studio
sends the nearest pause marker.

A text of one or two words is rendered with the 0.5 s marker and a filler sentence, as
[Short texts](features.md#short-texts) describes.

## Literal angle-bracket text

Angle-bracket strings such as `Hello <laugh/> there` and `<shouting>hello</shouting>` are ordinary text: they
get no special prompt, guidance, normalization protection or editing exemption. Pronunciation groups and the
start and pause markers keep their meaning. Every character counts in the character-rate steering schedule, and
reading-position spans include all overlapping non-whitespace tokens. Speech editing tokenizes and verifies the
enclosed words normally, so `Hello <laugh/> there` becomes `HELLO`, `LAUGH`, `THERE`, and a supplied
`original_text` keeps its angle-bracket strings.

## Steering controls

A direction is a measured vector that moves a voice along one property, such as an emotion, a speaking style, an
accent or pitch. A control applies one direction at a dose, and the LM's weights stay as they are. A set groups
directions, and the steering bundle lists the sets that a server loads.

The CUDA server takes `--steer-bundle FILE`. In `servers/sgl/scripts/serve.sh` that is `STEER_BUNDLE`, which
defaults to `assets/steering/studio_bundle_mect60k_x10.json`; an empty value serves no controls. The ggml, MLX
and in-browser servers read the same bundle converted to `steer.gguf` by `tools/convert_gguf.py steer`.

A request steers with these fields, and `GET /v2/controls` lists the sets and directions.

|Field|Meaning|
|---|---|
|`controls`|`{"<set>.<direction>": dose}`, applied to the whole utterance|
|`control_spans`|hidden controls over character ranges of the text|
|`control_span_rate`|frames per character, which places the spans; default 1|
|`control_span_follow`, `control_span_lead`, `control_span_lag`|place the spans by reading position; see [Span steering by reading position](#span-steering-by-reading-position)|
|`controls_cfg_scale`|guidance on the hidden controls; default 1, which is off|
|`steer_hold`, `steer_onset_seconds`, `steer_ramp_seconds`|release of a dose over a long line; see [Steer then hold](#steer-then-hold)|

```bash
curl -X POST http://localhost:1919/v2/generate \
  -H "Content-Type: application/json" \
  -d '{
        "text": "I cannot believe you did that!",
        "controls": {"mood.valence": 3.2, "mood.arousal": 3.2},
        "stream": true
      }' \
  --output excited.pcm
```

A `controls_cfg_scale` above 1 runs a second pass without the request's hidden controls and mixes the two as
`uncond + s · (cond − uncond)`, at about twice the compute. The second pass runs only for
`speaker_mode: embedding` requests with a speaker and a hidden control. A `controls_cfg_scale` other than 1 is a
400 together with `laci` or with a `speaker_cfg_scale` other than 1.

The shipped directions were fitted on `mect_60k`, the LM that preceded the release LM; doses can act
differently on other checkpoints.

### Steer then hold

A constant hidden push accumulates over a long line: the LM copies the style of the speech it has already
generated, so pitch and loudness climb. The bundle's `steer_release` block counters this. A request's hidden
`controls` run at full dose for `onset_seconds` of generated speech, ramp linearly to `hold` × dose over
`ramp_seconds`, and then stay at `hold`.

The shipped defaults are an onset of 4 s, a ramp of 1 s and a hold of 0.35, with `delivery.tempo` and
`delivery.whisper` exempt. Exempt directions, `control_spans`, raw directions and the tempo dose of a LACI take
stay constant.

`steer_hold` (0 to 1), `steer_onset_seconds` and `steer_ramp_seconds` override the defaults on `/v2/generate`,
`/v1/audio/speech` and `/v2/edit`; the two durations take 0 to 600 s. `steer_hold: 1` is a constant push, and
so is every dose under a bundle without `steer_release`. A `steer_hold` below 1 is a 400 on a server without a
bundle or whose bundle has no `steer_release`. `GET /v2/capabilities` and `GET /v2/controls` report the
defaults as `steer_release`, which is `null` without them.

[Steer then hold: schedule and measurements](tts_architecture.md#steer-then-hold-schedule-and-measurements)
gives the per-frame weights and the evaluation behind the defaults.

### Direction kinds

Each set directory has a `manifest.json` that names the set's space, and the space decides what the set's
directions act on.

- A raw set has `"space": "raw"`. Its directions move the speaker embedding, with the norm kept, and need
  `speaker_mode: embedding`; a raw control in reference mode is a 400.
- A hidden set has `"space": "attnres_source"` and `"source_index": k`. Its directions act on the hidden state
  of generated frames, and work in both speaker modes and in `control_spans`.

Reference and text rows are never steered. With no control requested, output equals that of a server without a
bundle. Doses are raw multiples of the measured shift, and the usable window is narrow: a style that responds at
one dose can run away at twice that dose.

### Set calibration

A set directory can hold `calibration.json`, per-direction doses chosen offline:

```json
{"default": {"<name>": 1.0}, "by_speaker": {"<speaker_id>": {"<name>": 1.0}}, "global_default": 1.0}
```

A `"<name>-"` key is the dose of the negative pole. The server never applies a calibration, so `controls` stay
raw doses. Clients read it in two places:

- `GET /v2/controls` has `sets[<set>].calibration` for a set with a calibration, with the keys `default`,
  `by_speaker` and `global_default`.
- Speaker objects of `GET /v2/speakers` have `disabled_controls`: the control ids whose calibrated dose is 0
  for that voice, in bundle order. An id is `"<set>.<name>"`, or `"<set>.<name>-"` when only the negative pole
  is 0. Without a bundle the list is `[]`.

The shipped `mood` set has seed doses and no per-voice entries, so `disabled_controls` is `[]` for every shipped
voice. The seed doses are 4 for valence, arousal, dominance, tempo and articulation, and 2 for loudness and
clarity.

### Shipped direction sets

The shipped bundle `assets/steering/studio_bundle_mect60k_x10.json` loads 14 sets.

|Set|Directory under `assets/steering/`|Kind|Directions|
|---|---|---|---|
|`identity`|`speaker_attributes_mect60k_x10`|raw|`gender`, `age`|
|`pitch`|`pitch_mect60k_x10`|raw|`pitch`, `pitch_range`|
|`accents_en`|`english_accents_mect60k_x10`|raw|`british`, `american`, `scottish`, `irish`, `australian`, `indian`, `welsh`|
|`accents_l1`|`speaker_accents_mect60k_x10`|raw|`hin`, `spa`, `fra`, `por`, `deu`, `vie`, `pol`|
|`mandarin`|`mandarin_mect60k_x10`|raw|`cmn`|
|`mandarin_hidden`|`mandarin_hidden_mect60k_x10`|hidden, source 15|`cmn`|
|`texture`|`voice_quality_mect60k_x10`|raw|`deep`, `shrill`, `husky`, `raspy`, `soft`, `authoritative`|
|`room`|`channel_mect60k_x10`|raw|`clean`|
|`delivery`|`ears_styles_mect60k_x10`|hidden, source 19|`whisper`, `loud`, `tempo`, `pitch`, `conversational`|
|`expressive`|`expressive_mect60k_x10`|hidden, source 19|`expressive`|
|`disfluency`|`disfluency_mect60k_x10`|hidden, source 19|`disfluent`|
|`mood`|`emotion_directions_mect60k_axes_x10`|hidden, source 19|`valence`, `arousal`, `dominance`, `loudness`, `tempo`, `clarity`, `articulation`|
|`emotions`|`ears_moods_mect60k_x10`|hidden, source 19|23 emotions, named under the table|
|`emphasis_nv_nodur`|`emphasis_nv45_nodur`|hidden, source 19|`emphasis`|

`emotions` holds angry, sad, happy, fearful, disgusted, surprised, adoring, confused, content, cute, desirous,
disappointed, distressed, embarrassed, ecstatic, guilty, interested, pained, proud, realizing, relieved, serene
and scared.

These controls have a measured dose range.

|Control|Usable doses|What to expect|
|---|---|---|
|`texture.*`|up to about 4|a dose of 2 barely moves the voice|
|`pitch.pitch`, `pitch.pitch_range`|−2 to 2|dose 1 goes from a low-pitched to a high-pitched speaker of the same gender; gender holds at −2 and 2; `pitch_range` leaves the F0 level about where it was|
|`room.clean`|−2 to 4|positive doses also widen the bandwidth and slow speech slightly|
|`delivery.conversational`|−4 to 4|4 sounds spontaneous: more pauses and prolongations, less enunciation, louder and flatter; −4 is the careful, quieter reading voice|
|`delivery.whisper`|up to 2|lowers voicing and level; 2 is strong and 4 breaks speech|
|`delivery.tempo`|−2 to 2|pace follows the dose on every voice; level and F0 rise with speed and fall with slowness; slow doses add pauses; −4 collapses|
|`expressive.expressive`|−16 to 16|livelier readings that are also higher and slower; `pitch.pitch_range` widens the F0 spread at a constant level and is the cleaner control|
|`disfluency.disfluent`|up to 8|hesitation as pauses, blocks and drawn-out words, without inserted fillers; past 8, words start to drop|
|`emotions.angry`, `emotions.happy`|2 to 3|the studio moods Angry and Happy|
|`emotions.sad`|4 to 6|the studio mood Sad|
|`emotions.serene`|4|the studio mood Calm|
|`emotions.scared`|8|the studio mood Scared|
|`emotions.confused`|2|the studio mood Confused|

The emotion doses are for the release LM. A negative dose takes an emotion out of a reference voice. Most of the
other 17 `emotions` directions add general emotional intensity and arousal more than their named emotion.
`delivery.loud` raises the level with the pitch rise of projected speech.

The studio's **Emphasis** control paints `emphasis_nv_nodur.emphasis` on one word. One unit of the control is a
dose of 4 of that direction plus 1 of `mood.arousal`, and its default is 4.5. At 4.5 the word's F0 peak rises
about 6 semitones, the word lengthens about 65 ms and it gets 1.6 dB louder. The bundle marks the control
`span_only`, with a `span_gate` of lead 0 and lag 0 for clients that send per-span reading gates.

The accent sets behave as follows:

- In `accents_l1`, strong doses trade voice identity for accent, and the speaker partly becomes a different one.
- `accents_en` directions are relative to American English. `american` is the reverse of `british` and is
  meant for British voices, and `welsh` also slows speech slightly.
- For a Mandarin accent, use `mandarin.cmn` and `mandarin_hidden.cmn` together: the two add up, at the price
  of the voice.
- No direction gives a Korean or Japanese accent; they need an accent adapter.

[How the shipped direction sets were fitted](tts_architecture.md#how-the-shipped-direction-sets-were-fitted)
names the data behind each set.

### Sets outside the steering bundle

The shipped bundle does not load these sets of `assets/steering/`.

|Directory|Kind|Directions|
|---|---|---|
|`accent_directions_mect60k`|hidden, source 15|`hin`|
|`speaker_emotions_mect60k`|raw|`angry`, `disgusted`, `fearful`, `happy`, `sad`, `surprised`|
|`ears_emotions_mect60k`|hidden, source 19|the same six emotions|
|`speaker_traits_mect60k`|raw|every speaker-embedding direction of the shipped bundle in one set, with `presets.json`|

Of the two emotion sets, the raw one moves the perceived emotion most, at a larger identity cost and in
embedding mode only. `presets.json` holds six persona presets as doses of the set's directions.

|Preset|Controls|
|---|---|
|elderly Scottish narrator|age 1.5, scottish 1.5, deep 2, pitch_range −1|
|young Australian|age −1.5, australian 1.5, pitch_range 1|
|gravelly old man|gender −1, age 2, raspy 3, deep 2|
|BBC newsreader|british 1.5, authoritative 3, clean 2|
|soft-spoken Irish woman|gender 1, irish 1.5, soft 3|
|Indian English caller|indian 1.5, clean −2|

The components add up, except the Scottish and Irish accents, which weaken in a mix. A persona is a different
speaker by design.

To serve a bundle of your own, list its set directories in a JSON file and pass the file as `STEER_BUNDLE`. Set
paths resolve relative to the bundle file.

```bash
cat > my_bundle.json <<'EOF'
{"name": "personas", "sets": {"traits": "assets/steering/speaker_traits_mect60k",
                              "delivery": "assets/steering/ears_styles_mect60k_x10"}}
EOF
STEER_BUNDLE=my_bundle.json servers/sgl/scripts/serve.sh
```

## Zonos Voice Studio

Every server serves the committed build `studio/dist/`: a chooser at `/`, the studio at `/studio` and the lite
page at `/lite`. [Zonos Voice Studio (UI)](../studio/README.md) documents the studio itself.

The studio's live steering controls need a steering bundle. `assets/steering/studio_bundle_mect60k_x10.json`
holds 14 sets, 25 UI controls, 15 accents, 8 personas, Who-map positions of the default voices and Mood-pad
landmarks of the six studio moods.

```bash
# STEER_BUNDLE defaults to assets/steering/studio_bundle_mect60k_x10.json.
servers/sgl/scripts/serve.sh
# Then open http://127.0.0.1:1919/studio
```

`GET /v2/controls` returns the sets, the directions with their raw vectors and the metadata of the studio's
controls, so a client can compute a steered speaker embedding itself. `GET /v2/speakers/{id}/embedding` returns
a voice's speaker embedding, and `POST /v2/speakers` accepts embedding files on an LM with both speaker modes.

With a bundle, the CUDA server embeds every default voice at startup. Its `serve.sh` then runs the speaker
encoder on the GPU: it sets `ZONOS3_SPEAKER_EMBEDDER_DEVICE=cuda` unless the variable is set.

### Span steering by reading position

By default, `control_spans` are placed by rate: a span starts at its character position × `control_span_rate`,
stretched by the bundle's `timing` factors, and ramps over 3 frames. With the same seed, the audio before a
span is identical to the audio without the span. `control_span_follow: "reading"` places the spans by where
the LM is reading.

Reading placement needs `--laci-heads` for the request's language; without heads, the rate schedule runs. On a
request with spans, the `X-Control-Span-Follow` response header says `reading` or `rate`. Reading placement
works with speaker CFG and with `laci=true`.

A span ramps in once the reading position is `control_span_lead` tokens before the span's first token; the
default is 1. It ramps out once the reading position is `control_span_lag` tokens past the first token after
the span; the default is 0.

Lead 1 places the span onset close to the spoken words. Lead 2 steers more of the span's words and leaks more
into its neighbors; lag −1 loses coverage and lag 1 leaks. Whisper and anger carry on past a span by
themselves, so their offsets tend to run late.
[Span steering by reading position](tts_architecture.md#span-steering-by-reading-position) describes the
mechanism.

### Projects on the server

Studio projects can live on the server. They are then shared by every browser that opens the studio link, and
they survive server restarts and a changed tunnel URL. `--studio-projects-dir` turns this on; the serve scripts
pass `ZONOS3_STUDIO_PROJECTS`, which is empty by default. With an empty value, projects stay in each browser's
IndexedDB.

The directory holds three kinds of file:

- `projects/<id>.json` is `{id, name, revision, updated_at, media_bytes, data}`, where `data` is the studio's
  project document.
- `media/<sha256>` is a content-addressed media file, written once.
- `media/<sha256>.type` holds that file's MIME type.

Project writes and garbage collection take a lock that servers sharing the directory honor, and a media upload
is a temp file renamed into place. `--studio-max-project-bytes` sets the largest project document, 20 MiB by
default, and `--studio-max-media-bytes` sets the largest media file, 2 GiB by default.

```bash
ZONOS3_STUDIO_PROJECTS=<projects dir> STEER_BUNDLE=assets/steering/studio_bundle_mect60k_x10.json \
  servers/sgl/scripts/serve.sh
```

The server then answers these routes.

|Route|Behavior|
|---|---|
|`GET /v2/projects`|returns `[{id, name, updated_at, revision, bytes, media_bytes, summary}]`, newest first; `summary` is the document's `data.summary` and `updated_at` is epoch milliseconds|
|`GET /v2/projects/{id}`|returns `{id, name, revision, updated_at, media_bytes, data}`|
|`PUT /v2/projects/{id}`|takes `{name, data, base_revision}`; returns `{id, revision, updated_at}`|
|`DELETE /v2/projects/{id}`|deletes the project|
|`POST /v2/projects/gc[?min_age_seconds=3600]`|removes media that no project lists and that are older than the grace period; returns `{removed, freed_bytes, kept}`|
|`HEAD`, `GET /v2/media/{sha256}`|streams the file with its stored type, `ETag: "<sha256>"` and `Cache-Control`; `If-None-Match` gives 304|
|`PUT /v2/media/{sha256}`|takes raw bytes, with the type from `?type=` or `Content-Type`; 201 when stored, 200 when already there|
|`GET /v2/capabilities`|has `projects: {enabled, max_media_bytes, max_project_bytes}`|

On `PUT /v2/projects/{id}`, `base_revision` must be the stored revision, or `null` for an id that is not
stored. Otherwise the answer is 409 `{detail, revision}` with the stored revision, where `null` means deleted,
so two tabs never overwrite each other silently. A project's media are the keys of `data.media`, which is
`{<sha256>: {type, size}}`. All of them must be stored, else the answer is 400 `{detail, missing: [sha256...]}`.

`{sha256}` is lowercase hex. `PUT /v2/media/{sha256}` hashes the upload while it streams to a temp file: a hash
mismatch is 400 `{detail, sha256: <actual>}`, and a file over the limit is 413. When the file is already
stored, the server answers 200 without reading the body. `Cache-Control` is
`public, max-age=31536000, immutable`, with `private` in place of `public` behind `ZONOS3_ACCESS_TOKEN`.
Garbage collection has a grace period because the uploads of a save in flight are newer than any project that
lists them.

## Plugins

A plugin adds routes to a server without changes to the server's code. `--plugins NAME...` imports the package
`zonos3_<NAME>` and mounts the routes of its `plugin()`; the ggml server also takes `NAME,NAME`, and the serve
scripts take `PLUGINS="NAME ..."`. Without `--plugins`, nothing is imported. `GET /v2/capabilities` lists the
mounted plugins in `plugins`. The `dub` plugin, a dubbing tool, lives in `Zyphra/ZONOS3-dub`, which is not
public.

Plugin `N` serves its API at `/v2/N` and `/v2/N/*` and its page at `/N` and `/N/*`. Its routes are mounted
ahead of the studio's, and `X-Keepalive` works on the POST paths that it lists.

A plugin sees the server through the host protocol of `zonos3.server.plugins`, whose version is `HOST_API`.

|Member|Purpose|
|---|---|
|`facts()`|checkpoint name, speaking-rate edges and buckets, LACI languages, speaker modes, raw controls, phoneme tokens, the bundle's `rate_extension`, the workers' device|
|`asr`|the server's Qwen3-ASR worker|
|`aligner`|the server's FuseAlign worker|
|`synthesize_laci(takes, session_id)`|runs LACI takes with a duration target|
|`encode_reference(audio, sample_rate, max_seconds)`|returns the reference codes of mono float32 audio|

The CUDA and MLX servers mount plugins in-process. The ggml server forwards plugin paths to its Python host,
which runs in the `--worker-python` interpreter, so the plugin must be installed there.

`synthesize_laci` runs `LaciTakeRequest` objects. Each is a `/v2/generate` request with `laci=true` plus a
`DurationTarget`, a target length in frames. With `adjust_rate`, a take that ends too early or too late is
retaken in the speaking-rate bucket that its duration ratio asks for, as
[LACI (alignment-guarded decoding)](features.md#laci-alignment-guarded-decoding) describes. A take can also run
on an extended speaking-rate scale: `RateBuckets` describes the scale, and the take carries its bucket and
doses in `rate_bucket` and `rate_doses`.

LACI needs `--laci-heads`. In the serve scripts, `LACI_ALL=1` passes the heads of every language from
`assets/laci_heads/`: `zonos3_mini_heads_<speech code>.json` for each non-English language and
`zonos3_mini_heads.json` for `en_us` and `en_gb`. `LACI_HEADS_DIR` and `LACI_HEADS` override the directory and
the list.

### Extended speaking-rate buckets (`rate_extension`)

The bundle's `rate_extension` block adds speaking-rate buckets below bucket 0 and above the row's last bucket.
Each added bucket is the speaking-rate row left unknown plus a `delivery.tempo` dose, and a language has only
the buckets that were measured safe for it.

The shipped block was measured on a 10-bucket speaking-rate row, while the release LM `lm/zonos3_mini` has 30
buckets. Servers do not check the block against the LM: `facts().rate_extension` is the block as the bundle has
it. A plugin must therefore compare the block's `row_edges` with `facts().rate_edges` and use the row's own
buckets when they differ.

The block has steered buckets for nine languages.

|Language|Steered buckets|
|---|---|
|en|−7 to −1, 10 to 12|
|fr|−4 to −1, 10 to 11|
|de|−5 to −1|
|es|−4 to −1, 10|
|it|−4 to −1|
|pt|−3 to −1, 10|
|ja|−5 to −1|
|zh|−6 to −1|
|ko|−2 to −1|

Reference-mode voices use only the faster steered buckets, which the block states as
`reference_sides: ["faster"]`. On the slower side, reference-mode voices lost words or garbled speech, and the
buckets were not validated for them. Embedding-mode voices use both sides.
[Extended speaking-rate buckets: derivation](tts_architecture.md#extended-speaking-rate-buckets-derivation)
describes how the edges and doses were derived.
