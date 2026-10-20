# Hosting

## Headers for threads

WASM threads need `SharedArrayBuffer`, which browsers enable only on
cross-origin isolated pages:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp      # or: credentialless
```

With `require-corp`, every cross-origin resource the page loads must send CORS
or `Cross-Origin-Resource-Policy: cross-origin`. Hugging Face and jsDelivr send
CORS headers, and the runtime fetches them in CORS mode. Without isolation,
everything still works on 1 thread, and `load()` reports `actualThreads: 1`.

Examples: `tools/serve.mjs` (local), the Vite configs in `examples/`, and for
Netlify/Cloudflare Pages a `_headers` file:

```
/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
```

These headers affect third-party embeds on your page (iframes, images, fonts
without CORS). Test your whole app, not only the TTS part. This package never
changes your app's headers.

## Self-hosting assets

| Option | What to serve | Notes |
|---|---|---|
| `modelBaseUrl` | `bundle.json`, `tokenizer.model`, `bos_before_voice.npy` and the `.onnx` files for your precision | `node tools/fetch-models.mjs --precision int8-wide` downloads and verifies them. Hashes are always checked |
| `voiceBaseUrl` | `voices/alba-casual.emb.npy` | `node tools/prepare-stock-voice.mjs`. Defaults to `modelBaseUrl` |
| `ortBaseUrl` | `node_modules/onnxruntime-web/dist/` (needs `ort.wasm.min.mjs` and `ort-wasm-simd-threaded.{mjs,wasm}`) | Must be version 1.20.0. Serve `.wasm` as `application/wasm` |
| `workerUrl` | `dist/worker.js` | Only if your bundler can't handle `new Worker(new URL(...))` |

## Caching

Assets are stored in Cache Storage under `pocket-tts:<language>@<revision>`.
A new model revision gets a new cache; on load, older revisions of the same
language are deleted. `PocketTTS.clearCache()` removes all of them. A full
quota only disables caching; it never fails a load.
