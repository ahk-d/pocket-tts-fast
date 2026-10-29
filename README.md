# pocket-tts-fast

Fast, local [Kyutai Pocket TTS](https://huggingface.co/kyutai/pocket-tts) in
the browser. Speech is generated on the user's device with onnxruntime-web
(WebAssembly) in a Web Worker. Text and voice clips are not uploaded anywhere.

* **Dynamic-cache optimization with matching PCM in tested cases.** The stock graphs
  copy a fixed 1000-slot KV cache, 49 MB, into and out of WASM on every frame.
  This runtime sizes the caches to what is actually used. The output is
  bit-identical in the tested German and English WASM cases. See [How it's faster](#how-its-faster).
* **Stock voice first.** The Alba voice loads as a 0.5 MB embedding, and no
  voice encoder is needed.
* **Optional local cloning.** Clips are encoded on the device, once. The
  resulting embedding can be reused across the supported precision profiles.
* **Sentence-ending guards.** Every sentence is generated as its own chunk
  with its own end-of-speech tail. Ending-corpus checks pass; listening
  validation and the limitations below still apply. A sentence
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

Status: **alpha**. German (default) and English are validated, on browser WASM
(CPU) only. Not an official Kyutai project.

## Quick start

```bash
npm install pocket-tts-fast@alpha
```

Copy `node_modules/pocket-tts-fast/voices/` into your app's static
directory (for example, `public/voices/` in Vite). Serve the file at
`/voices/alba-casual.emb.npy` (for English: `/voices/alba-casual.en.emb.npy`)
and retain its attribution README. The embedding
is included in npm; no separate upload or preparation script is needed.
Set `voiceBaseUrl` to the directory **containing** `voices/`, not to `voices/`
itself. For an app hosted at the domain root, use `/`:

```js
import { PocketTTS } from "pocket-tts-fast";

const tts = new PocketTTS({ voiceBaseUrl: "/" }); // see "Assets" below
await tts.load((e) => e.type === "progress" && console.log(e.label, e.loaded, e.total));
const voice = await tts.useVoice("alba");

// Generate and play (queues 1 s of audio, then plays while generating).
const { metrics, played } = await tts.speak("Guten Morgen! Wie geht es dir?", { voice });
await played;

// Or just get the audio: mono Float32 chunks at 24 kHz.
const chunks = [];
await tts.generate("Bis später.", { voice, onChunk: (pcm) => chunks.push(pcm) });

// English: same API, its own model files and voice embedding.
const en = new PocketTTS({ language: "english", voiceBaseUrl: "/" });
```

Generation is client-only: create `PocketTTS` in the browser (for example in an
effect or event handler). Importing the module during SSR does nothing.

### Cloning a voice (optional, local)

```js
import { decodeAudioFile } from "pocket-tts-fast";

const { audio, sampleRate } = await decodeAudioFile(file); // decoded in the browser
const embedding = await tts.encodeVoice(audio, { inputSampleRate: sampleRate }); // loads the ~40 MB encoder once
const clone = await tts.loadVoiceEmbedding(embedding); // store `embedding` to skip encoding next time
```

Only use recordings you have the right to use, with the speaker's consent.
Review the [upstream model-use conditions](https://huggingface.co/kyutai/pocket-tts)
as well as the model and voice licenses.

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
`GERMAN`, `ENGLISH`, `MANIFESTS` (the asset manifests).

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

Installing npm includes the small Alba embedding; no ONNX models or extra
assets are fetched by install scripts. At `load()` the runtime fetches:

* **Graphs, tokenizer, metadata:** from the commit-pinned Hugging Face revision
  of [KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx)
  (`58a6d00`). Each file is checked against the SHA-256 and size in
  [`src/manifest.ts`](https://github.com/ahk-d/pocket-tts-fast/blob/main/src/manifest.ts); a mismatch is an error, never a
  fallback. Override `modelBaseUrl` to self-host.
* **onnxruntime-web 1.20.0:** from jsDelivr (pinned version). Override
  `ortBaseUrl` to self-host `node_modules/onnxruntime-web/dist/`.
* **Stock voice:** `voices/alba-casual.emb.npy` (0.5 MB). It is included in this npm package under `voices/`, licensed separately
  under CC BY 4.0. Copy it to your app's public/static folder as above;
  fetching it uses `voiceBaseUrl`. Large model files are not bundled.
  See [hosting instructions](https://github.com/ahk-d/pocket-tts-fast/blob/main/docs/hosting.md).

## How it's faster

Pocket TTS runs FlowLM once per 80 ms audio frame. The exported graphs declare
six attention caches of `[2, 1, 1000, 16, 64]` float32. onnxruntime-web copies
every input into the WASM heap and every output back out, so each frame
handles **49 MB of cache inputs and 49 MB of cache outputs**, though
a short sentence uses far fewer slots. These are tensor byte sizes, not
profiled memory-bus throughput.
The graphs only read `Slice[0 : step + len]`, so the unused slots are never
needed.

This runtime makes the cache axis symbolic. The change is made in memory when
the stock graphs load, and only the declared shapes change, not weights or
nodes. The runtime then passes caches holding just the filled prefix plus
headroom, and grows them when needed. In the Mimi decoder, the passthrough
(`Identity`) caches become empty and the ring caches stay at 1000.

* State passed into FlowLM per call: **49.15 MB → 9.44 MB**; into the decoder:
  **16.49 MB → 8.30 MB** (short prompts).
* PCM is **bit-identical** to the fixed-cache path in every case tested
  (German and English, onnxruntime-web 1.20.0 WASM), and to the original
  runtime in the German reference cases.
* Speed (German only; English is not benchmarked yet): about **2× faster** in
  a labelled smoke run (RTF 1.6–1.8 → 0.84–0.89,
  live gaps 3–8 → 0). Setup: Intel Core i7-9750H, Node 22, 4 WASM threads,
  INT8-wide, stock voice, machine under load. Earlier development measurements
  on the same machine: RTF ≈ 1.1 → 0.6 in Node with 4 threads, and 6.6–10.6 → 1.2
  in a 1-thread browser. Controlled, per-device release numbers are still to
  come ([protocol](https://github.com/ahk-d/pocket-tts-fast/blob/main/bench/protocol.md)).

Details, evidence and caveats: [docs/cache-optimization.md](https://github.com/ahk-d/pocket-tts-fast/blob/main/docs/cache-optimization.md).

## Development

The following commands require a source checkout; tools, tests and examples
are in the GitHub repository, not the npm tarball.

```bash
git clone https://github.com/ahk-d/pocket-tts-fast.git
cd pocket-tts-fast
npm install
npm test                 # build + 86 unit tests (no models needed)
npm run test:types       # public .d.ts against a usage file
npm run test:consumers   # npm pack → install the tarball into vanilla / Vite / React-Vite examples → build
node tools/fetch-models.mjs --precision int8-wide --cloning   # pinned + verified download into ./models
node tools/prepare-stock-voice.mjs                           # Alba embedding (CC BY 4.0 source, hash-checked)
npm run test:models      # real models: fixed vs dynamic PCM identity, cache growth, sentence endings on a 14-item corpus
POCKET_TTS_WAV_DIR=out npm run test:models && python3 tools/check_transcripts.py out test/models/endings.de.json   # Whisper check
npm run serve            # demo at http://localhost:8787 (cross-origin isolated)
node bench/bench.mjs --cache both --repeat 5                 # see bench/protocol.md
```

### Comparison page (GitHub Pages or Vercel)

`demo/compare/` runs the same phrases with the fixed and the dynamic cache in
one browser tab and reports speed, first audio and whether the PCM matches.
`npm run build:site` builds it into `site/` (comparison at the root, full demo
at `demo/`) with relative paths, the pinned onnxruntime-web files and the stock
voices; model graphs load from the pinned Hugging Face revision.

* **GitHub Pages:** `.github/workflows/pages.yml` deploys on every push to
  `main` (one-time: Settings → Pages → Source: GitHub Actions). Pages cannot set
  headers, so [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker)
  (MIT) adds the cross-origin isolation headers after one automatic reload.
* **Vercel:** `vercel.json` sets the headers directly; import the repository.

Examples: [`examples/vanilla`](https://github.com/ahk-d/pocket-tts-fast/blob/main/examples/vanilla) (no bundler),
[`examples/vite-vanilla`](https://github.com/ahk-d/pocket-tts-fast/blob/main/examples/vite-vanilla),
[`examples/react-vite`](https://github.com/ahk-d/pocket-tts-fast/blob/main/examples/react-vite) (includes a small `usePocketTTS`
hook). Vue, Svelte and Next.js have not been tested.

## Known limitations

* **German and English only.** English uses the `english_2026-04` bundle at
  the same pinned revision. Other language folders in the model repo are not
  in the manifest and are not validated.
* **English: a trailing ellipsis ("I'm not sure…") can end loud.** The voice
  keeps going past end-of-speech, so that sentence's last sound is cut. Whisper
  still heard every word. English also keeps 3 frames after end-of-speech (5
  for chunks of up to 4 words) instead of German's 1 (3), which adds up to
  0.16 s of tail per sentence.
* **Numbers are read as written.** "3.15 Uhr" is not expanded to words, and
  Whisper heard it as "3, 5 Uhr". Write numbers out if exact reading matters.
* **Uppercase umlauts (Ä Ö Ü), é and € are not in the tokenizer vocabulary.**
  They fall back to byte tokens; they were not seen to cause problems, but
  they are untested beyond the ending corpus. Typographic quotes, apostrophes,
  dashes and ellipses are normalized automatically, because they did cause
  problems.
* **An utterance-initial "Ich" was missed by Whisper (small)** in 2 of 14
  corpus items ("Ich komme…" was heard as "Komm…"). Speech starts within 50 ms,
  but onset timing alone cannot establish that every word is intact. This
  remains a known alpha limitation requiring human listening.
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
Model weights are downloaded at runtime. The bundled Alba embedding is
CC BY 4.0, not MIT; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Built by Ali
with Codex and Claude; the dynamic-cache change was implemented by Claude and
verified in Ali's sessions.
