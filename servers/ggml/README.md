# zonos3.cpp

The ggml server `zonos3-server` and the command-line tool `zonos3-cli` are C++ programs that run
[ZONOS3 Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) from GGUF files on CPU, CUDA, Metal or Vulkan.
Synthesis and reference encoding need no Python; conversion, speech editing and plugins do. The build needs a
Rust toolchain for the text normalizer.

Related documents:

- [About ZONOS3-Mini](../../README.md#about-zonos3-mini) describes the model.
- [API reference](../../docs/api.md) lists the HTTP routes and request fields.
- [ZONOS3 feature guide](../../docs/guide.md) describes voice cloning, speech editing and steering.
- [zonos3.cpp internals](docs/INTERNALS.md) explains the implementation.

## Build

The build needs:

- CMake 3.18 or later and a C++17 compiler
- `cargo`, which builds the text normalizer
- the submodules `servers/ggml/ggml` and `third_party/zyphra-textnorm`

```bash
# From the repo root.
git submodule update --init servers/ggml/ggml third_party/zyphra-textnorm
cd servers/ggml
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CUDA:
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
```

CMake options select the backend and the desktop app.

|CMake option|Backend|
|---|---|
|none|CPU; Metal as well on Apple platforms|
|`-DGGML_CUDA=ON`|CUDA|
|`-DGGML_VULKAN=ON`|Vulkan|
|`-DZONOS3_APP=ON`|also builds the desktop app `zonos3-app`: CMake 3.31 or later, a C++23 compiler, the platform webview (Linux: GTK4, WebKitGTK 6.0, libadwaita, json-glib)|

Metal, Vulkan and Windows builds depend on the installed toolchain.

## Models

The files are the `gguf/` folder of [Zyphra/ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini). Run the
commands of this section from `servers/ggml`.

|File|Size|Needed for|
|---|---|---|
|`backbone-ptq.gguf`|0.66 GB|everything: the LM, with Q4_0 experts and Q8_0 mixers|
|`zac-keep2.gguf`|0.45 GB|everything: the default decoder (`keep2_plain`)|
|`spk-encoder.gguf`|45 MB|cloning from audio in embedding mode|
|`zac-encoder.gguf`, `ssl-teacher.gguf`|4.03 GB, 1.55 GB|cloning from audio in reference mode; speech editing|
|`zac.gguf`|2.20 GB|the full-size decoder (`full_mr`)|
|`steer.gguf`|0.2 MB|steering controls|
|`voices/`|22 MB|the default voices, with their reference codes and speaker embeddings|
|`backbone-ptq-edit.gguf`, `edit/accents.gguf`|0.66 GB, 0.26 GB|speech editing|

```bash
# The LM, the default decoder and the speaker encoder.
hf download Zyphra/ZONOS3-Mini gguf/backbone-ptq.gguf gguf/zac-keep2.gguf gguf/spk-encoder.gguf --local-dir .
# Reference mode and speech editing: the reference encoder.
hf download Zyphra/ZONOS3-Mini gguf/zac-encoder.gguf gguf/ssl-teacher.gguf --local-dir .
# Steering controls.
hf download Zyphra/ZONOS3-Mini gguf/steer.gguf --local-dir .
# Default voices, for --tts-default-voices-dir.
hf download Zyphra/ZONOS3-Mini --include 'gguf/voices/*' --local-dir .
# Speech editing: the edit backbone and the speech-edit adapter.
hf download Zyphra/ZONOS3-Mini gguf/backbone-ptq-edit.gguf gguf/edit/accents.gguf --local-dir .
# Everything, 9.9 GB, with the full-size decoder zac.gguf.
hf download Zyphra/ZONOS3-Mini --include 'gguf/*' --local-dir .
```

- When their flags are omitted, the server and the CLI load `zac-keep2.gguf`, `zac.gguf`, `spk-encoder.gguf`,
  `zac-encoder.gguf`, `ssl-teacher.gguf`, `steer.gguf` and `accents.gguf` from the backbone's directory.
- Leave `edit/accents.gguf` in `edit/`: it is the speech-edit adapter and needs `backbone-ptq-edit.gguf`, so
  next to `backbone-ptq.gguf` it is loaded and refused.
- A GGUF without the metadata this build requires is refused at load; download or convert it again.

### Convert from the PyTorch weights

Download the PyTorch weights of [Zyphra/ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) or use a local
copy.

```bash
R=release
hf download Zyphra/ZONOS3-Mini --include 'lm/*' --include 'autoencoder/*' --include 'speaker_encoders/*' \
  --include 'registries/*' --local-dir "$R"

# Run from servers/ggml.
T="uv run --package zonos3-tools python ../../tools/convert_gguf.py"
mkdir -p gguf
# F16 reference backbone.
$T lm "$R/lm/zonos3_mini" --outtype f16 -o gguf/backbone-f16.gguf
# Default backbone: Q4_0 experts and Q8_0 mixers, quantized after training.
$T lm "$R/lm/zonos3_mini" --outtype q4_0-exp-q8_0-mix -o gguf/backbone-ptq.gguf
$T codec-decoder "$R/autoencoder/generator.pt" --head "$R/autoencoder/heads/keep2_plain" -o gguf/zac-keep2.gguf
$T speaker-encoder "$R/speaker_encoders/ownmect_s2/ckpt_0037.pth" -o gguf/spk-encoder.gguf

# Optional: the reference encoder, for reference mode and speech editing.
$T codec-encoder "$R/autoencoder/generator.pt" --outtype f32 -o gguf/zac-encoder.gguf
$T ssl-teacher "$R/autoencoder/teacher" --outtype f32 -o gguf/ssl-teacher.gguf

# Optional: the steering bundle of assets/steering/.
$T steer ../../assets/steering/studio_bundle_mect60k_x10.json -o gguf/steer.gguf

# Optional: the full-size decoder full_mr, 32/12/12/12/12 layers per stage.
$T codec-decoder "$R/autoencoder/generator.pt" --head "$R/autoencoder/heads/full_mr" -o gguf/zac.gguf

# Optional: speech editing (the edit backbone and its adapter).
$T lm "$R/lm/zonos3_mini" --outtype q4_0-exp-q8_0-mix \
  --edit-adapter "$R/lm/zonos3_mini/edit_lora/z3m_omni3b_edit_lora.pt" \
  --phoneme-rows "$R/lm/zonos3_mini/text_rows/pause_tokens_s5000.pt" \
  -o gguf/backbone-ptq-edit.gguf
mkdir -p gguf/edit && $T accents "$R/lm/zonos3_mini/edit_lora" -o gguf/edit/accents.gguf
```

`tools/convert_gguf.py` has one subcommand per file and loads each checkpoint strictly, so a missing or unknown
tensor is an error. The backbone file embeds the tokenizer.
[Convert the release](docs/INTERNALS.md#convert-the-release) documents each subcommand.

## Run the server

The launcher `start-zonos3.sh` is for release archives. It starts `zonos3-server`, the binary that you can also
run directly.

### Launcher

```bash
ZONOS3_SERVER_BIN="$PWD/build/zonos3-server" ZONOS3_MODEL_DIR="$PWD/gguf" \
  bash ./scripts/start-zonos3.sh -y --no-browser
```

Release archives ship the server next to `start-zonos3.sh` (Windows: `start-zonos3.bat`). The launcher reads
models from `models/` next to the script unless `ZONOS3_MODEL_DIR` is set, and it never downloads models.

The launcher requires the backbone file and the default decoder, and passes `spk-encoder.gguf`,
`zac-encoder.gguf` and `ssl-teacher.gguf` to the server when they exist. Without `spk-encoder.gguf` it prints a
note, because cloning from audio in embedding mode is then off. When the server is ready, the launcher opens
`http://127.0.0.1:1919/`.

ffmpeg is used only to decode reference audio for cloning. When no ffmpeg is in `$ZONOS3_MODEL_DIR/bin` or on
`PATH`, the launcher offers to download one into that directory; `-y` skips the question.

A launcher flag overrides its environment variable.

|Flag|Env|Default|Meaning|
|---|---|---|---|
|`--quant Q`|`ZONOS3_QUANT`|`ptq`|serve `backbone-Q.gguf`; `--quant f16` serves the F16 reference backbone|
|`--zac Z`|`ZONOS3_ZAC`|`zac-keep2`|default decoder `Z.gguf`; when the other of `zac-keep2.gguf` and `zac.gguf` is present, it is loaded too|
|`--cpu`, `--gpu`|none|`cpu` (`gpu` in the Vulkan and Metal archives)|backend|
|`--host H`|`ZONOS3_HOST`|`127.0.0.1`|bind address|
|`--port P`|`ZONOS3_PORT`|`1919`|port|
|`-y`, `--yes`|`ZONOS3_ASSUME_YES=1`|off|do not ask before fetching ffmpeg|
|`--no-browser`|`ZONOS3_NO_BROWSER=1`|off|do not open a browser|
|`-- ARGS`|none|none|pass `ARGS` to `zonos3-server`|

`ZONOS3_SERVER_BIN` names the server binary; its default is `zonos3-server` next to the script.

The desktop app `zonos3-app` has file pickers for the backbone file, one decoder and an optional speaker
encoder, and like the launcher it never downloads models.

### zonos3-server

```bash
./build/zonos3-server gguf/backbone-ptq.gguf \
  --tts-default-voices-dir ../../assets/default_voices --studio-projects-dir ~/.zonos3/projects --port 1919
curl -fsS http://127.0.0.1:1919/v2/capabilities
curl -fsS http://127.0.0.1:1919/v2/generate \
  -d '{"text":"Hello.","stream":false,"format":"wav","seed":1}' -o hello.wav
# Streaming returns mono float32 PCM at 44.1 kHz; X-Seed echoes the seed.
curl -fsSN http://127.0.0.1:1919/v2/generate \
  -d '{"text":"Streaming hello.","stream":true,"seed":1}' -o hello.pcm
# The full-size decoder zac.gguf; X-Decoder names the decoder used.
curl -fsS http://127.0.0.1:1919/v2/generate \
  -d '{"text":"Hello.","stream":false,"format":"wav","seed":1,"decoder":"full_mr"}' -o hello-full.wav
```

`zonos3-server` serves the HTTP API of [`servers/api/openapi.yaml`](../api/openapi.yaml): the routes under
`/v2/` and the OpenAI-compatible `GET /v1/models` and `POST /v1/audio/speech`. It also serves the studio build:
a chooser at `/`, Zonos Voice Studio at `/studio` and the lite page at `/lite`.

The first argument is the backbone file, and a file flag that is omitted falls back to the file of that name
next to it.

|Flag|Default|Meaning|
|---|---|---|
|`--zac PATH`|`zac-keep2.gguf` and, when present, `zac.gguf`|a decoder; repeatable, and the first is the default|
|`--spk-encoder PATH`|`spk-encoder.gguf`|speaker encoder, for cloning from audio in embedding mode|
|`--zac-encoder PATH`, `--ssl-teacher PATH`|`zac-encoder.gguf`, `ssl-teacher.gguf`|reference encoder and its SSL teacher, for reference mode and speech editing|
|`--host H`|`127.0.0.1`|bind address|
|`--port P`|`1919`|port; 0 picks a free port|
|`--gpu`, `--cpu`|GPU on Apple Silicon, CPU elsewhere|backend|
|`--zac-cpu`|off|run the decoders on the CPU even with `--gpu`|
|`--zac-gpu`|off|run the decoders on the GPU even with `--cpu`|
|`--max N`|2000|frames per request, about 133 s; a long prompt lowers it to the rows the slot has left|
|`--batch N`|8|concurrent syntheses; memory grows with it|
|`--zac-threads N`|4|decode lanes per decoder|
|`--reference-idle-unload N`|60|free the reference encoder and the SSL teacher after `N` idle seconds; 0 keeps them loaded|
|`--stream-decode-default MODE`|`prefix`|streaming decode of requests without `stream_decode`: `prefix` sends chunks as frames arrive, `whole` sends one chunk at the end|
|`--tts-default-voices-dir DIR`|`default_voices`, relative to the working directory|default voices; an empty value turns them off|
|`--python-src DIR`|`model/src` of the checkout that holds the executable, else `python/` next to it|directory of the `zonos3` Python package|
|`--worker-python PY`|none|interpreter of the Python host, which runs speech editing and plugins|
|`--plugins NAME[,NAME...]`|none|plugins to load; repeatable|
|`--steer PATH`|`steer.gguf`|steering bundle|
|`--accent-adapters PATH`|`accents.gguf`|accent adapters; the speech-edit adapter|
|`--laci-heads LANG=PATH`|none|LACI alignment heads of one language; repeatable|
|`--tts-phoneme-lexicon DIR`|none|pronunciation lexicon for `/v2/phonemize`|
|`--studio-projects-dir DIR`|none|studio projects and media on the server|
|`--studio-dir DIR`|`studio/dist` of the checkout, else `studio/` next to the executable|studio build|

`--gpu` takes a discrete GPU, else an integrated one, else the CPU, and the startup line
`zonos3: backend = ...` names the device it took.

The server reads these environment variables.

|Variable|Default|Meaning|
|---|---|---|
|`ZONOS3_ACCESS_TOKEN`|none|token that every request must carry; see [Headers](../../docs/api.md#headers)|
|`ZONOS3_FFMPEG`|`ffmpeg` next to the executable, else on `PATH`|ffmpeg binary that decodes uploaded audio|
|`ZONOS3_PREFIX_STREAM_FIRST_CHUNK_FRAMES`|1|frames in the first streamed chunk; each later chunk doubles|
|`ZONOS3_PREFIX_STREAM_CHUNK_FRAMES`|15|largest streamed chunk, in frames|
|`ZONOS3_PREFIX_STREAM_HOLD_BACK_FRAMES`|0|complete frames held back until the end of the request|
|`ZONOS3_ASR_PYTHON`|`<repo>/.venv-asr/bin/python`|interpreter of the speech recognition worker|
|`ZONOS3_ASR_DEVICE`|`mps` on macOS; else `cuda:0` when available, else the CPU|device of the speech recognition worker|
|`ZONOS3_FUSEALIGN_DEVICE`|`cuda:0` when available, else the CPU|device of the FuseAlign worker|
|`HF_HUB_OFFLINE`|1, set by the Python host unless the variable is already set|0 lets the Python host download from Hugging Face|

## Server behavior

Request fields and status codes are in [API reference](../../docs/api.md); the subsections below hold what
differs on this server. Text normalization is built in, and a request turns it off with
`text_normalization: false`.

### Limits

- Speaker CFG and controls CFG need `--batch` 2 or more.
- `max_tokens` is at most `--max`.
- A prompt that fills the slot is a 400 `TTS input len {rows} exceeds {capacity}`.
- Reference codes keep their first 900 frames (60 s).
- Early partial-frame streaming needs a decoder GGUF with per-codebook `res.{i}.bias` tensors; without them
  the server streams complete frames and says so at startup.

### Decoders

Each `--zac` loads one decoder, and the first is the default. A decoder's id is its GGUF's
`mossdual_istft.head.variant`, which is `keep2_plain` or `full_mr`; the same id twice is a startup error.
`GET /v2/capabilities` lists the loaded decoders in `decoders`, the default first.

A request selects a decoder with `decoder` and the response names it in `X-Decoder`; an id that the server did
not load is a 400. When a request leaves `headroom_db` unset, the selected decoder's own headroom applies.

### Voices

`--tts-default-voices-dir` takes a directory of audio files and `.npy` or `.npz` speaker embeddings. The
server's default `speaker_mode` is `reference`, while `zonos3-cli --clone` defaults to embedding mode;
[Voice cloning](../../docs/guide.md#voice-cloning) defines the two modes.

For an audio voice, a sibling `<stem>.codes.npy` or `<stem>.embedding.npy` is used and the encoder does not run,
so regenerate both files after you change a voice:

```bash
V=../../assets/default_voices/NAME
./build/zac-encoder-cli gguf/zac-encoder.gguf gguf/ssl-teacher.gguf --audio "$V.mp3" "$V.codes.npy" --gpu
./build/spk-encoder-cli gguf/spk-encoder.gguf --audio "$V.mp3" "$V.embedding.npy" --cpu
```

The encoders load on first use, and the server caches each voice's speaker embedding and reference codes.

### Speech editing

1. Start the edit backbone with its adapter:

   ```bash
   ./build/zonos3-server gguf/backbone-ptq-edit.gguf \
     --accent-adapters gguf/edit/accents.gguf --worker-python <PY>
   ```

2. Put `zac-encoder.gguf`, `ssl-teacher.gguf` and `spk-encoder.gguf` next to the backbone file.
3. Give `--worker-python` an interpreter `PY` that has the workspace packages. When you run these commands
   from the repo root, the last one prints such an interpreter:

   ```bash
   git submodule update --init third_party/fusealign-multilingual
   uv sync
   uv run which python
   ```

4. Put the FuseAlign checkpoint (1.34 GB) in the Hugging Face cache, because the Python host runs with
   `HF_HUB_OFFLINE=1` unless the variable is already set and so does not download it. Either run the command
   below, or start the server once with `HF_HUB_OFFLINE=0`.

   ```bash
   hf download Zyphra/fusealign fusealign-multilingual-cjkft-step2500.safetensors
   ```

5. Optional: for speech recognition, run `bash tools/setup_asr_env.sh` from the repo root, which creates
   `.venv-asr`. Without it `/v2/capabilities` reports `edit_transcribe: false`, and the requests that need a
   transcript from the server answer 503.

Release archives ship `python/zonos3` and `python/fusealign_hub` (Apache-2.0), so `PY` there needs only their
dependencies: `torch`, `torchaudio`, `transformers`, `safetensors`, `huggingface_hub` and `silero-vad`.
Installing FuseAlign with pip brings all six:

```bash
pip install git+https://github.com/Zyphra/fusealign-multilingual.git
```

### Plugins

`--plugins NAME[,NAME...]` loads the package `zonos3_<NAME>` in the `--worker-python` interpreter, so install
it there with `uv pip install --python PY --no-deps -e <plugin checkout>`; the server checks the import at
startup. [Plugins](../../docs/guide.md#plugins) describes the URL rule and the host protocol.

### Steering

A `steer.gguf` next to the backbone file, or `--steer PATH`, turns on steering controls. Build it from the
steering bundle:

```bash
uv run --package zonos3-tools python ../../tools/convert_gguf.py steer \
  ../../assets/steering/studio_bundle_mect60k_x10.json -o gguf/steer.gguf
```

[Steering controls](../../docs/guide.md#steering-controls) lists the request fields, and
[Steer then hold](../../docs/guide.md#steer-then-hold) describes how a dose is released over a long line.
On this server, `control_span_follow: "reading"` and `laci: true` need `--laci-heads LANG=PATH` for the
request's language.

The shipped directions were fitted on `mect_60k`, the LM that preceded the release LM; doses can act
differently on other checkpoints.

## zonos3-cli

```bash
# Decoder is discovered beside the backbone.
./build/zonos3-cli gguf/backbone-ptq.gguf --tts 'Hello, world.' out.wav --seed 1

# Clone from audio in embedding mode (the default) and save the embedding.
./build/zonos3-cli gguf/backbone-ptq.gguf --tts 'Hello again.' clone.wav \
  --clone ../../assets/default_voices/AmericanFemale.mp3 --save-speaker-embedding voice-emb.npy --seed 1

# Reuse a saved embedding without loading the speaker encoder.
./build/zonos3-cli gguf/backbone-ptq.gguf --tts 'A cached embedding.' cached.wav \
  --speaker-embedding voice-emb.npy --condition-buckets 4,3,10 --seed 1

# Reference mode: encode once, save the codes, reuse them.
./build/zonos3-cli gguf/backbone-ptq.gguf --tts 'Raw reference.' ref.wav \
  --clone ../../assets/default_voices/AmericanFemale.mp3 --speaker-mode reference \
  --save-reference voice-codes.npy --seed 1
./build/zonos3-cli gguf/backbone-ptq.gguf --tts 'Cached codes.' ref2.wav \
  --reference-codes voice-codes.npy --seed 1

# Speaker CFG takes two slots and needs a speaker source.
./build/zonos3-cli gguf/backbone-ptq.gguf --tts 'Guided.' cfg.wav \
  --speaker-embedding voice-emb.npy --cfg-scale 1.5 --seed 1
```

`zonos3-cli` decodes with one decoder, and like the server it looks next to the backbone file for the file of
an omitted flag.

|Flag|Default|Meaning|
|---|---|---|
|`--gpu`, `--cpu`|GPU on Apple Silicon, CPU elsewhere|backend|
|`--clone <audio>`|none|encode reference audio|
|`--speaker-mode MODE`|`embedding`|how `--clone` conditions the prompt: `embedding` or `reference`|
|`--speaker-embedding <emb.npy>`|none|saved float32 `[192]` speaker embedding; excludes `--clone` and `--reference-codes`|
|`--reference-codes <codes.npy>`|none|saved C-order `[frames,10]` integer codes in `0..4095`; frames past 900 (60 s) are dropped|
|`--save-speaker-embedding <out.npy>`, `--save-reference <out.npy>`|none|write the computed speaker embedding or reference codes|
|`--loudness <LUFS>`, `--fmax <Hz>`, `--speaking-rate <bytes/s>`|unknown|condition values, mapped to buckets by the LM's edges|
|`--condition-buckets a,b,c`|unknown|bucket ids `0..n` for loudness, fmax and speaking rate, where `n` means unknown; a bucket id wins over a value|
|`--cfg-scale <s>`|1 (off)|speaker CFG scale; needs a speaker source|
|`--silence-prefill <n>`|3|codec-silence frames after the text, `0..30`; their audio is never emitted|
|`--headroom-db <db>`|3, the decoder's value|output attenuation, `0..24`; 0 is the raw decoder level|
|`--zac <zac.gguf>`|`zac-keep2.gguf`|the decoder|
|`--spk-encoder`, `--zac-encoder`, `--ssl-teacher`|`spk-encoder.gguf`, `zac-encoder.gguf`, `ssl-teacher.gguf`|encoder files; in `zonos3-cli` they need `--clone`|
|`--steer <steer.gguf>`|`steer.gguf`|steering bundle, loaded when a control flag is given|
|`--control ID=DOSE`|none|one control at a dose; repeatable; the ids are those of `GET /v2/controls`|
|`--control-span START:END:ID=DOSE[,ID=DOSE...]`|none|hidden controls over code points `[START,END)` of the text, ramped over 0.2 s; repeatable|
|`--control-span-rate R`|1|frames per character, which places the spans|
|`--steer-hold H`, `--steer-onset S`, `--steer-ramp S`|the `steer_release` of `steer.gguf`|steer then hold: full dose for `--steer-onset` seconds, a ramp over `--steer-ramp` seconds to `H` times the dose|
|`--accent NAME`|none|accent adapter to apply; needs `--tts` or `--generate`|
|`--accents PATH`|`accents.gguf`|file that holds the accent adapters|
|`--accent-strength S`|1|scale of the accent adapter|
|`--max N`|450|limit of generated rows, including the rows that drain the codebook delay|
|`--seed N`|0|sampling seed|

`--reference-codes` takes ordinary codes as `zac-encoder-cli` writes them, not the delayed rows that the decoder
consumes. With `--cfg-scale` the logits are `uncond + s·(cond − uncond)`, where the unconditioned pass has no
speaker source and unknown buckets.

A raw control moves the speaker embedding and works in embedding mode only, while a hidden control acts on
every generated frame in both modes. Span positions are scaled by the timing factors of the request's
`--control` flags. A `steer.gguf` without `steer_release` keeps every dose constant, and `--steer-hold` below 1
is then an error.

To clone from an audio file, install ffmpeg or set `ZONOS3_FFMPEG`. The prompt and the generated rows together
must fit the 32768-position context.

## Other tools

```bash
# The first command also writes ids.cond.json.
./build/zonos3-cli gguf/backbone-ptq.gguf --build-prompt 'Hello.' ids.npy
./build/zac-cli gguf/zac-keep2.gguf codes.npy decoded.wav
./build/spk-encoder-cli gguf/spk-encoder.gguf --audio voice.mp3 voice-emb.npy --cpu
./build/zac-encoder-cli gguf/zac-encoder.gguf gguf/ssl-teacher.gguf \
  --audio voice.mp3 voice-codes.npy --cpu
./build/quantize-cli gguf/backbone-f16.gguf gguf/backbone-q8_0.gguf q8_0
./build/quantize-cli gguf/backbone-f16.gguf gguf/backbone-q6_k-exp.gguf \
  q6_k --experts-only --spine-f16
```

A prompt dump has a `.cond.json` file next to the ids, with the speaker embedding, the condition buckets and
the row layout. `zac-cli` reads `<codes>.eos.npy` (`[eos]` or `[eos, prefill]`) and takes `--skip-frames N`
and `--headroom-db X`.

`backbone-ptq.gguf` is the release LM quantized after training. Other `quantize-cli` profiles have not been
evaluated on this model; measurements are in [Quantization](docs/INTERNALS.md#quantization).

## Test

```bash
ctest --test-dir build --output-on-failure
bash tests/readme-commands.sh build/zonos3-cli build/zonos3-server
bash tests/launcher-smoke.sh build/zonos3-server
```

The fixtures are random-weight models that test interfaces, not speech quality. Numerical checks against
PyTorch are in [Reference comparison](docs/INTERNALS.md#reference-comparison); the optional SSL-teacher
regression is in [Smoke tests](docs/INTERNALS.md#smoke-tests).
