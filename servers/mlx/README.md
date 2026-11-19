# zonos3-mlx

The MLX server is the ZONOS3 Mini HTTP server for Apple Silicon. It serves the API of
[servers/api/openapi.yaml](../api/openapi.yaml) and Zonos Voice Studio, and runs every network in
[MLX](https://github.com/ml-explore/mlx) from the GGUF files of the ggml server.

[API reference](../../docs/api.md) lists the routes and request fields, and
[ZONOS3 feature guide](../../docs/guide.md) describes voice cloning, speech editing and steering.

## Install and run

```bash
uv sync --locked --package zonos3-mlx
hf download Zyphra/ZONOS3-Mini --include 'gguf/*' --local-dir servers/ggml   # servers/ggml/gguf/
servers/mlx/scripts/serve.sh            # 127.0.0.1:1919: / chooser, /studio, /lite
```

The server reads the GGUF files of the ggml server from `servers/ggml/gguf/`.
[Models](../ggml/README.md#models) lists their sizes and which files each feature needs.

## Files

The server loads these files.

|File|Used for|
|---|---|
|`backbone-ptq.gguf`|the LM (`lm/zonos3_mini`); `backbone-f16.gguf` also loads|
|`zac-keep2.gguf`, `zac.gguf`|the decoders `keep2_plain` (default) and `full_mr`|
|`spk-encoder.gguf`|cloning from audio in embedding mode|
|`zac-encoder.gguf`, `ssl-teacher.gguf`|cloning from audio in reference mode; speech editing|
|`steer.gguf`|steering controls|
|`accents.gguf`|accent adapters; the speech-edit adapter|

Every file is found next to the backbone file unless its flag names another path. `--zac` is repeatable, and
the first decoder is the default. A decoder's id is its GGUF's `mossdual_istft.head.variant`, and two files with
one id are refused. The reference encoder loads on first use. [API reference](../../docs/api.md) describes the
`decoder` request field and the `X-Decoder` response header.

## Configuration

`serve.sh` takes the variables of [Server configuration](../../README.md#server-configuration) except
`STEER_BUNDLE`, because this server reads `steer.gguf`. It appends extra flags to `python -m zonos3_mlx`.

`serve.sh` and the server also read these variables.

|Variable|Default|Meaning|
|---|---|---|
|`MODELS_DIR`|`servers/ggml/gguf`|directory that holds the GGUF files|
|`SERVE_MODEL`|`$MODELS_DIR/backbone-ptq.gguf`|the backbone file; on this server it is a GGUF|
|`WORKER_PYTHON`|the server's interpreter|interpreter of the FuseAlign worker; passes `--worker-python`|
|`ZONOS3_STREAM_DECODE_DEFAULT`|`prefix`|streaming decode of requests without `stream_decode`; `whole` decodes once at the end|
|`ZONOS3_TTS_NORM`|`1`|`0` turns text normalization off|
|`ZONOS3_FUSEALIGN_DEVICE`|`cuda:0` when available, else the CPU, so the CPU on macOS|torch device of the FuseAlign worker|
|`ZONOS3_ASR_PYTHON`|`.venv-asr/bin/python`|interpreter of the speech recognition worker|
|`ZONOS3_ASR_DEVICE`|`mps` on macOS|device of the speech recognition worker|
|`ZONOS3_FFMPEG`|`ffmpeg` on `PATH`|ffmpeg binary that decodes uploaded audio|
|`HF_HUB_OFFLINE`|1, set by the workers unless the variable is already set|0 lets the workers download from Hugging Face|

`python -m zonos3_mlx --help` lists every flag.

|Flag|Default|Meaning|
|---|---|---|
|`--zac`|`zac-keep2.gguf` and `zac.gguf` next to the backbone file|a decoder GGUF; repeatable|
|`--spk-encoder`|`spk-encoder.gguf`|speaker encoder, for cloning from audio in embedding mode|
|`--zac-encoder`|`zac-encoder.gguf`|reference encoder, for reference mode and speech editing; needs `--ssl-teacher`|
|`--ssl-teacher`|`ssl-teacher.gguf`|SSL teacher of the reference encoder|
|`--steer`|`steer.gguf`|steering bundle|
|`--accent-adapters`|`accents.gguf`|accent adapters; the speech-edit adapter|
|`--host`|`127.0.0.1`|bind address|
|`--port`|`1919`|port|
|`--max`|2000|frames per request, about 133 s|
|`--batch`|4|concurrent generation slots; a CFG request takes two|
|`--reference-idle-unload`|60|free the reference encoder and the SSL teacher after this many idle seconds; 0 keeps them|
|`--tts-default-voices-dir`|`assets/default_voices`|default voices: audio files and `.npy` or `.npz` speaker embeddings; an empty value turns them off|
|`--worker-python`|the server's interpreter|interpreter of the FuseAlign worker|
|`--laci-heads`|none|`LANG=PATH` LACI alignment heads; takes several|
|`--plugins`|none|plugin names; takes several|
|`--tts-phoneme-lexicon`|none|pronunciation lexicon for `/v2/phonemize`|
|`--studio-projects-dir`|none|studio projects and media on the server|
|`--studio-max-project-bytes`|20 MiB|largest project document|
|`--studio-max-media-bytes`|2 GiB|largest media file|
|`--studio-dir`|`studio/dist`|studio build|
|`--log-level`|`info`|log level|

## Optional features

Plain synthesis needs none of the features below.

### Speech editing

Speech editing needs the edit backbone with the speech-edit adapter, the reference encoder files and FuseAlign.
Start the server on the edit backbone:

```bash
SERVE_MODEL=servers/ggml/gguf/backbone-ptq-edit.gguf servers/mlx/scripts/serve.sh \
  --accent-adapters servers/ggml/gguf/edit/accents.gguf
```

FuseAlign places every word of a clip. Install it and download its checkpoint before the first edit:

```bash
git submodule update --init third_party/fusealign-multilingual
uv sync --locked --package zonos3-mlx
hf download Zyphra/fusealign fusealign-multilingual-cjkft-step2500.safetensors
```

The FuseAlign worker runs with `HF_HUB_OFFLINE=1` unless the variable is already set, so the checkpoint must be
in the Hugging Face cache before the first alignment.
[Speech editing](../../docs/guide.md#speech-editing) describes the endpoints and request fields.

### Speech recognition

Speech recognition is optional. `bash tools/setup_asr_env.sh` creates `.venv-asr` with Qwen3-ASR, which lets
the server transcribe clips sent without their transcript and run `verify`. Without it, `/v2/capabilities`
reports `edit_transcribe: false`.

### Plugins

`--plugins NAME...`, or `PLUGINS="..."` for `serve.sh`, imports the package `zonos3_<NAME>` from the server's
interpreter and mounts its routes under `/v2/<NAME>` and `/<NAME>`. A plugin shares the server's speech
recognition and FuseAlign workers. The `dub` plugin lives in `Zyphra/ZONOS3-dub`, which is not public.
[Plugins](../../docs/guide.md#plugins) describes the URL rule and the host protocol.

## Tools

`python -m zonos3_mlx.tools` runs one network or one synthesis without the HTTP server.

|Command|Does|
|---|---|
|`python -m zonos3_mlx.tools decode`|decodes delayed codes to 44.1 kHz PCM before the output gain, as an `.npy` file|
|`python -m zonos3_mlx.tools embed`|turns audio into a speaker embedding|
|`python -m zonos3_mlx.tools ssl`|writes the SSL teacher's features and hidden states|
|`python -m zonos3_mlx.tools encode`|turns audio into `[frames, 10]` reference codes|
|`python -m zonos3_mlx.tools generate`|runs one synthesis in process|
|`--cpu`|runs MLX on the CPU device; applies to every `tools` command|
|`python -m zonos3_mlx.validate`|dumps one LM forward's captures for `tools/compare_golden.py`|

## Test

```bash
uv run --package zonos3-mlx pytest servers/mlx/tests                       # tiny-model parity, decoder, server
uv run --package zonos3-api pytest servers/api/tests/conformance --base-url http://127.0.0.1:1919
```

## Implementation notes

The notes below describe how this server differs from the ggml server, whose GGUF files it reads.

### Scheduler

One worker thread owns the LM. Up to `--batch` requests advance one row per tick; a CFG request takes two slots
and runs both forwards in one MLX evaluation. Q4_0 and Q8_0 blocks load as MLX affine-quantized weights without
requantization.

Sampling runs on the GPU with the reference sampler's random stream, so a seed reproduces the reference rows up
to float rounding. When no request reads the alignment probe, which LACI and reading-position spans do, the
next row's forward is queued before the host reads the sample back.

Audio decodes on a second thread while the LM continues.
[Streaming decode](../../docs/tts_architecture.md#streaming-decode) describes prefix decoding and early
partial-frame decode, which a request sets with `stream_codebooks` and `stream_codebooks_scope`. A decoder GGUF
without per-codebook `res.{i}.bias` tensors streams complete frames.

### Parity with the reference

The goldens come from `tools/dump_golden.py`.

|Network|Agreement|
|---|---|
|F16 backbone, `backbone-f16.gguf`|logits cosine 1.000000, 100 % argmax agreement|
|quantized backbone with its Q8_0 logits head, against the same model with its F16 head|cosine at least 0.99996 in prefill and 0.999999 in decode; argmax agreement 98.4 % and 100 %|
|SSL teacher on `--cpu`|cosine 1.000000|
|speaker encoder on `--cpu`|cosine 1.000000|
|reference codes of the release voices, against the committed `<stem>.codes.npy`|97-98 % of the first codebook, 67-96 % of the residual codebooks; the decoded clips transcribe identically|

The quantized backbone was compared on `full_ids.npy` of the in-browser server's goldens. Its logits head is
quantized to Q8_0 at load, which makes decode steps about 43 % faster on an M-series GPU.

In the SSL teacher, a few dozen of about 800,000 elements in three late layers fall outside the tolerances of
atol 1e-4 and rtol 1e-3. The residual codebooks agree less than the first because RVQ residuals amplify BF16
rounding differences.
