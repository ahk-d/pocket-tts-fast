# @flowent/pocket-tts

Fast, local [Kyutai Pocket TTS](https://huggingface.co/kyutai/pocket-tts) in
the browser. Speech is generated on the user's device with onnxruntime-web
(WebAssembly) in a Web Worker. Text and voice clips are not uploaded anywhere.

* **About 2× faster generation on this machine, same audio.** The stock graphs
  copy a fixed 1000-slot KV cache, 49 MB, into and out of WASM on every frame.
  This runtime sizes the caches to what is actually used. The output is
  bit-identical. See [How it's faster](#how-its-faster).
* **Stock voice first.** The Alba voice loads as a 0.5 MB embedding, and no
  voice encoder is needed.
* **Optional local cloning.** Clips are encoded on the device, once. The
  resulting embedding works with every precision.
* **Clean sentence endings.** Every sentence is generated as its own chunk
  with its own end-of-speech tail, so the last word isn't clipped. A sentence
  scanner keeps "Dr.", "z. B.", "3.14" and "3. Oktober" intact, and
  typographic quotes, apostrophes and dashes are normalized (left in, they
  derailed the model). Checked on real generated speech with Whisper.
* **Streaming playback.** Audio arrives in 240 ms blocks. You can queue 1 s
  before playing (the default), start immediately, start adaptively when the
  measured speed predicts no gaps, or wait for the whole recording.
* **Framework-independent.** Plain ES modules with TypeScript types, no
  runtime dependencies and no side effects on import (SSR-safe). Works from a
  `<script type="module">` and in bundlers that understand
  `new Worker(new URL(..., import.meta.url))` (Vite, webpack 5).

Status: **alpha**. German is the only validated language, on browser WASM
(CPU) only. Not an official Kyutai project.

## Quick start

```bash
npm install @flowent/pocket-tts
```

```js
import { PocketTTS } from "@flowent/pocket-tts";

const tts = new PocketTTS({ voiceBaseUrl: "/voices" }); // see "Assets" below
await tts.load((e) => e.type === "progress" && console.log(e.label, e.loaded, e.total));
const voice = await tts.useVoice("alba");

// Generate and play (queues 1 s of audio, then plays while generating).
const { metrics, played } = await tts.speak("Guten Morgen! Wie geht es dir?", { voice });
await played;

// Or just get the audio: mono Float32 chunks at 24 kHz.
const chunks = [];
await tts.generate("Bis später.", { voice, onChunk: (pcm) => chunks.push(pcm) });
```

Generation is client-only: create `PocketTTS` in the browser (for example in an
effect or event handler). Importing the module during SSR does nothing.

### Cloning a voice (optional, local)

```js
import { decodeAudioFile } from "@flowent/pocket-tts";

const { audio, sampleRate } = await decodeAudioFile(file); // decoded in the browser
const embedding = await tts.encodeVoice(audio, { inputSampleRate: sampleRate }); // loads the ~40 MB encoder once
const clone = await tts.loadVoiceEmbedding(embedding); // store `embedding` to skip encoding next time
```

Only use recordings you have the right to use.

## API

| | |
|---|---|
| `new PocketTTS(options)` | `precision`: `"int8-wide"` (default, 125 MB), `"int8-core"` (173 MB) or `"fp32"` (400 MB). Also `modelBaseUrl`, `voiceBaseUrl`, `ortBaseUrl`, `maxThreads` (default 4), `cache` (Cache Storage, default on), `dynamicCache` (default on), `temperature` (0.3), `seed` (42) |
| `load(onEvent?)` | Downloads or reads from cache, verifies SHA-256, creates sessions. Concurrent calls share one load. Resolves to `{ modelId, precision, actualThreads, requestedThreads, crossOriginIsolated, loadedFiles, … }` |
| `useVoice(id)` | Prepares a stock voice; switching voices never reloads the model |
| `encodeVoice` / `loadVoiceEmbedding` / `cloneVoice` | Local cloning |
| `generate(text, { voice, onChunk, deliveryFrames })` | Streams `Float32Array` chunks. `meta.isSilence` marks the 0.25 s pause inserted between sentences. Metrics include `complete` and per-sentence `textChunks` (`generatedFrames`, `emittedFrames`, `endReason`: `eos`, `cancelled`, `frame-budget` or `frame-cap`) |
| `speak(text, { voice, mode, primeSeconds, audioContext })` | Generate and play; returns `{ metrics, chunks, player, played }` |
| `stop()` / `destroy()` | Stop works mid-generation and while still buffering; `destroy` rejects pending calls |
| `PocketTTS.clearCache()` | Deletes cached model revisions |
| `chunksToWavBlob(chunks)` | 16-bit WAV |

Expert exports: `PocketTTSEngine` (the platform-agnostic generation loop),
`createWebHost`, `makeCacheDynamic`, `StreamPlayer`, `adaptiveStartSeconds`,
`GERMAN` (the asset manifest).

## Browser requirements

* WebAssembly SIMD and module workers: current Chrome, Edge, Firefox and Safari.
* **Threads need cross-origin isolation.** Serve your page with
  `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`). Without
  them the runtime uses 1 thread, which is several times slower. `load()`
  reports `actualThreads`.
* About 125 MB of graphs for INT8-wide on first visit, plus the ~11 MB
  onnxruntime-web WASM binary. Later visits read from Cache Storage, keyed by
  model revision.

## Assets

Nothing is downloaded at `npm install`. At `load()` the runtime fetches:

* **Graphs, tokenizer, metadata:** from the commit-pinned Hugging Face revision
  of [KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx)
  (`58a6d00`). Each file is checked against the SHA-256 and size in
  [`src/manifest.ts`](src/manifest.ts); a mismatch is an error, never a
  fallback. Override `modelBaseUrl` to self-host.
* **onnxruntime-web 1.20.0:** from jsDelivr (pinned version). Override
  `ortBaseUrl` to self-host `node_modules/onnxruntime-web/dist/`.
* **Stock voice:** `voices/alba-casual.emb.npy` (0.5 MB). It is **not hosted
  publicly yet**: generate it with `node tools/prepare-stock-voice.mjs` and
  serve it under `voiceBaseUrl`. See [docs/hosting.md](docs/hosting.md).

## How it's faster

Pocket TTS runs FlowLM once per 80 ms audio frame. The exported graphs declare
six attention caches of `[2, 1, 1000, 16, 64]` float32. onnxruntime-web copies
every input into the WASM heap and every output back out, so each frame
moves **49 MB in and 49 MB out**, though a sentence uses only ~100–200 slots.
The graphs only read `Slice[0 : step + len]`, so the unused slots are never
needed.

This runtime makes the cache axis symbolic. The change is made in memory when
the stock graphs load, and only the declared shapes change, not weights or
nodes. The runtime then passes caches holding just the filled prefix plus
headroom, and grows them when needed. In the Mimi decoder, the passthrough
(`Identity`) caches become empty and the ring caches stay at 1000.

* State passed into FlowLM per call: **49.15 MB → 9.44 MB**; into the decoder:
  **16.49 MB → 8.30 MB** (short prompts).
* PCM is **bit-identical** to the fixed-cache path and to the original runtime
  in every case tested (German, onnxruntime-web 1.20.0 WASM).
* Speed: about **2× faster** in a labelled smoke run (RTF 1.6–1.8 → 0.84–0.89,
  live gaps 3–8 → 0). Setup: Intel Core i7-9750H, Node 22, 4 WASM threads,
  INT8-wide, stock voice, machine under load. Earlier development measurements
  on the same machine: RTF ≈ 1.1 → 0.6 in Node with 4 threads, and 6.6–10.6 → 1.2
  in a 1-thread browser. Controlled, per-device release numbers are still to
  come ([protocol](bench/protocol.md)).

Details, evidence and caveats: [docs/cache-optimization.md](docs/cache-optimization.md).

## Development

```bash
npm install
npm test                 # build + 85 unit tests (no models needed)
npm run test:types       # public .d.ts against a usage file
npm run test:consumers   # npm pack → install the tarball into vanilla / Vite / React-Vite examples → build
node tools/fetch-models.mjs --precision int8-wide --cloning   # pinned + verified download into ./models
node tools/prepare-stock-voice.mjs                           # Alba embedding (CC BY 4.0 source, hash-checked)
npm run test:models      # real models: fixed vs dynamic PCM identity, cache growth, sentence endings on a 14-item corpus
POCKET_TTS_WAV_DIR=out npm run test:models && python3 tools/check_transcripts.py out test/models/endings.de.json   # Whisper check
npm run serve            # demo at http://localhost:8787 (cross-origin isolated)
node bench/bench.mjs --cache both --repeat 5                 # see bench/protocol.md
```

Examples: [`examples/vanilla`](examples/vanilla) (no bundler),
[`examples/vite-vanilla`](examples/vite-vanilla),
[`examples/react-vite`](examples/react-vite) (includes a small `usePocketTTS`
hook). Vue, Svelte and Next.js have not been tested.

## Known limitations

* **German only.** Other language folders in the model repo are not in the
  manifest and are not validated.
* **Numbers are read as written.** "3.15 Uhr" is not expanded to words, and
  Whisper heard it as "3, 5 Uhr". Write numbers out if exact reading matters.
* **Uppercase umlauts (Ä Ö Ü), é and € are not in the tokenizer vocabulary.**
  They fall back to byte tokens; they were not seen to cause problems, but
  they are untested beyond the ending corpus. Typographic quotes, apostrophes,
  dashes and ellipses are normalized automatically, because they did cause
  problems.
* **An utterance-initial "Ich" was missed by Whisper (small)** in 2 of 14
  corpus items ("Ich komme…" was heard as "Komm…"). Speech starts within 50 ms,
  so nothing is cut off; this still needs a human listening check.
* Browser WASM on CPU only. WebGPU, FP16, INT4 and React Native are not
  supported in this release.
* RTF depends heavily on the device, on threads and on what else is running.
  Measure on your target devices.

## License and credits

Code: MIT ([LICENSE](LICENSE)), derived in part from
[vlapky/pocket-tts-js](https://github.com/vlapky/pocket-tts-js) (MIT).
Model: Kyutai Pocket TTS, CC BY 4.0. ONNX export and INT8 graphs: KevinAHM,
CC BY 4.0. Voice "Alba": Alba MacKenna via
[kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices), CC BY 4.0.
Weights and voices are downloaded at runtime and are not covered by the MIT
license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Built by Ali
with Codex and Claude; the dynamic-cache change was implemented by Claude and
verified in Ali's sessions.
