# Benchmark protocol

Use this protocol before quoting any number in a release. Numbers from a busy
machine are labelled `stress` and never used as headline results.

## What is measured

| Metric | Definition |
|---|---|
| Generation (wall) | Request to last chunk received, for the complete utterance |
| Audio duration | Samples actually returned ÷ 24 000, including inserted pauses between text chunks |
| RTF | Generation ÷ audio duration (below 1 = faster than real time) |
| First chunk | Request to first PCM block (first 3 frames = 240 ms of audio) |
| Scheduled start (browser) | Request to the audio-clock time playback is scheduled (`startLatency`); not a speaker measurement |
| Live gaps | Chunks that arrive > 20 ms after the previous one finished playing, if playback starts on the first chunk. Natural pauses are generated audio and do not count |
| State MB per call | Bytes of KV/state tensors passed into FlowLM / the decoder per run (exact) |
| PCM SHA-256 | Over the Float32 samples. Must match between fixed and dynamic caches |

## Rules

* One model workload at a time on the device. `bench/bench.mjs` refuses to run
  if the 1-minute load average exceeds half the logical cores, unless
  `--allow-load` is passed, in which case the result is labelled `stress`.
* One complete warmup per configuration (precision × cache mode × voice).
* At least 5 repetitions. Prompt order rotates each repetition, and fixed and
  dynamic runs alternate, so neither gets a systematically warmer turn.
* Report medians **and** observed ranges per prompt. Report failures and
  stopped runs; never drop them.
* Record the CPU, OS, browser or Node version, onnxruntime-web version,
  requested **and actual** threads, cross-origin isolation, precision, voice,
  delivery frames and playback policy. A run without isolation is a 1-thread
  run, whatever was requested.
* Report cold first visit (network and session creation) and warm-cache
  readiness separately from generation.
* No composite score. If speedups differ by device or prompt, show the
  variation.

## Commands

```bash
# Controlled run (idle machine), fixed vs dynamic, stock voice, all prompts:
node bench/bench.mjs --precision int8-wide --threads 4 --cache both --repeat 5 --json bench/results/<date>/<device>/int8-wide-4t.json
# Each precision:
for p in fp32 int8-core int8-wide; do node bench/bench.mjs --precision $p --cache both --repeat 5 --json bench/results/<date>/<device>/$p.json; done
```

In the browser, use the demo with `npm run serve` (cross-origin isolated, so
threads work). Open the Laboratory section and export the session results as
JSON. The built-in browser pane of the Claude desktop app cannot run threaded
WASM; use Chrome, Edge, Firefox or Safari.

## Not yet done

* A browser automation runner (`run-browser.mjs`) and the published-npm
  comparison row (`pocket-tts-js@0.1.0`) from the release plan.
* The 12-input corpus from the release spec, which lives in the Codex
  workspace that was not readable here. `bench/prompts.de.json` holds the
  four prompts used during development.
* Listening and transcript panels per precision.
