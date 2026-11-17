---
title: ZONOS3 Mini
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12.12"
app_file: app.py
pinned: false
short_description: Multilingual TTS with voice cloning and steering controls
startup_duration_timeout: 1h
models:
  - Zyphra/ZONOS3-Mini
---

# ZONOS3 Mini

The Zonos Voice Studio (Cast, Direct, Edit and the lite page) over the plain-PyTorch reference
implementation of ZONOS3 Mini, with every take and every speech edit generated on ZeroGPU.
Pronunciations can be typed in X-SAMPA between double square brackets, "I met [[S@vQn]]" (the release phoneme tokens),
and pauses as `<0.5>` after a word. "Phonemize selection" looks words up in the gold pronunciation lexicon (`lexicon/`);
words it lacks are guessed with espeak-ng (GPL-3.0, through `espeakng-loader`). Projects stay in your browser. LACI
(Project settings, or Advanced on the lite page) regenerates a take from where it skipped words, babbled or stopped early.

The lexicon is built from WikiPron (Wiktionary, CC BY-SA), CMUdict, ipa-dict and Britfone; `lexicon/manifest.json`
lists every source file with its URL and SHA-256.

The first start downloads about 10.5 GB of weights, so it takes several minutes.

Source: `servers/space/` in Zyphra/ZONOS3, deployed with `servers/space/deploy.py`.
