# Third-party notices

This package includes one derived Alba voice embedding in voices/, licensed
under CC BY 4.0 with attribution in voices/README.md. It includes no source
recordings or ONNX model weights. Model weights download at runtime; their
licenses apply independently of this package's MIT code license.

## Code

### pocket-tts-js — MIT
Copyright (c) 2026 vlapky. <https://github.com/vlapky/pocket-tts-js>, source
pinned at commit `7d7a27423b0845eb0425c81a8aa5ed3f3d973eef`. The tokenizer,
.npy parser, text chunking, state handling and generation loop in `src/core/`
are derived from it. MIT license text: see LICENSE.

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
at https://github.com/ahk-d/pocket-tts-fast/blob/main/src/manifest.ts. The runtime rewrites only the declared shape of the
attention-cache inputs/outputs of `flow_lm_main` and `mimi_decoder` in memory
(see https://github.com/ahk-d/pocket-tts-fast/blob/main/docs/cache-optimization.md); weights and nodes are unchanged.

## Voices

**Alba** (stock voice): recording "casual.wav" voice-acted by Alba MacKenna,
from <https://huggingface.co/kyutai/tts-voices> (folder `alba-mackenna/`,
revision `323332d33f997de8394f24a193e1a76df720e01a`), released under CC BY 4.0
per that repository's README. The stock voice file is an encoder embedding
derived from the first 10 s of that recording by the preparation recipe at
https://github.com/ahk-d/pocket-tts-fast/blob/main/tools/prepare-stock-voice.mjs.
This 512128-byte embedding (SHA-256 e4549924b9054a02c33907431ebce573011837f68084785499c43df9945741e0)
is bundled in npm under voices/alba-casual.emb.npy; retain its attribution
and the CC BY 4.0 license link when redistributing it.
Other folders of kyutai/tts-voices carry different licenses (for example
Expresso is CC BY-NC 4.0); do not assume CC BY 4.0 for them.

Suggested attribution when you ship the stock voice:
"Speech synthesized with Kyutai Pocket TTS (CC BY 4.0). Voice: Alba MacKenna via kyutai/tts-voices (CC BY 4.0)."

## Upstream model-use conditions

Review the upstream model cards and their stated use conditions, including
consent for voice cloning:
https://huggingface.co/kyutai/pocket-tts
https://huggingface.co/KevinAHM/pocket-tts-onnx

This package does not waive those conditions or imply endorsement.

## coi-serviceworker (demo site only)

`demo/compare/coi-serviceworker.min.js` is coi-serviceworker v0.1.7 by Guido
Zuidhof and contributors, MIT License, https://github.com/gzuidhof/coi-serviceworker.
It is used only by the comparison page and is not part of the npm package.
