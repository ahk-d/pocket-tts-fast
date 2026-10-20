# models/

Local model folder used by the tools, tests, benchmark and demo server. It is
git-ignored except for this file. Nothing here is published to npm.

```
models/german/bundle.json, tokenizer.model, bos_before_voice.npy, *.onnx   ← node tools/fetch-models.mjs
models/german/voices/alba-casual.emb.npy                                     ← node tools/prepare-stock-voice.mjs
models/german/*_dyncache.onnx (optional)                                      ← python3 tools/make_dynamic_cache.py
```

`node tools/verify-assets.mjs` checks every file against `src/manifest.ts`.
