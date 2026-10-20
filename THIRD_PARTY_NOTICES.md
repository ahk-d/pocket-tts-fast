# Third-party notices

This package contains no model weights, voice recordings or voice embeddings.
It downloads them at runtime from the locations below; their licenses apply to
your use of them independently of this package's MIT license.

## Code

### pocket-tts-js — MIT
Copyright (c) 2026 vlapky. <https://github.com/vlapky/pocket-tts-js>, source
pinned at commit `7d7a27423b0845eb0425c81a8aa5ed3f3d973eef`. The tokenizer,
.npy parser, text chunking, state handling and generation loop in `src/core/`
are derived from it. MIT license text: see `LICENSE`.

### onnxruntime-web — MIT
Copyright (c) Microsoft Corporation. <https://github.com/microsoft/onnxruntime>.
Not bundled: the worker imports version 1.20.0 at runtime from `ortBaseUrl`
(default: jsDelivr), or from a copy you host.

## Models — CC BY 4.0

**Pocket TTS** by Kyutai, <https://huggingface.co/kyutai/pocket-tts>, licensed
under Creative Commons Attribution 4.0 International
(<https://creativecommons.org/licenses/by/4.0/>).

**ONNX export and INT8 graphs** by KevinAHM,
<https://huggingface.co/KevinAHM/pocket-tts-onnx> (CC BY 4.0), revision
`58a6d00cf13d239b6748cb0769f35c580a8f606c`. Files and SHA-256 hashes are pinned
in `src/manifest.ts`. The runtime rewrites only the declared shape of the
attention-cache inputs/outputs of `flow_lm_main` and `mimi_decoder` in memory
(see `docs/cache-optimization.md`); weights and nodes are unchanged.

## Voices

**Alba** (stock voice): recording "casual.wav" voice-acted by Alba MacKenna,
from <https://huggingface.co/kyutai/tts-voices> (folder `alba-mackenna/`,
revision `323332d33f997de8394f24a193e1a76df720e01a`), released under CC BY 4.0
per that repository's README. The stock voice file is an encoder embedding
derived from the first 10 s of that recording by `tools/prepare-stock-voice.mjs`.
Other folders of kyutai/tts-voices carry different licenses (for example
Expresso is CC BY-NC 4.0); do not assume CC BY 4.0 for them.

Suggested attribution when you ship the stock voice:
"Speech synthesized with Kyutai Pocket TTS (CC BY 4.0). Voice: Alba MacKenna via kyutai/tts-voices (CC BY 4.0)."
