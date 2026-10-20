# Dynamic KV caches: removing a 49 MB copy from every frame

## The problem

Pocket TTS generates audio autoregressively: one FlowLM run per 80 ms audio
frame, plus a Mimi decoder run per delivered block. Both graphs are stateful.
Their attention caches are graph inputs and outputs that the caller passes back
in on the next call.

The published ONNX export declares every attention cache with a fixed length of
**1000 slots**:

| Graph | Cache inputs | Declared shape | Bytes per call (in) |
|---|---|---|---|
| `flow_lm_main` | 6 self-attention caches | `[2, 1, 1000, 16, 64]` float32 | 6 × 8.19 MB = **49.15 MB** |
| `mimi_decoder` | 2 `encoder_transformer` caches | `[2, 1, 8, 1000, 64]` float32 | 2 × 4.10 MB |
| `mimi_decoder` | 2 `decoder_transformer` ring caches | `[2, 1, 8, 1000, 64]` float32 | 2 × 4.10 MB |

onnxruntime-web keeps tensors in JavaScript memory and copies every input into
the WASM heap before a run, then copies every output back out. With fixed
shapes, **each frame moves about 49 MB into WASM and 49 MB back out**, even
though a typical sentence fills only ~100–200 of the 1000 slots. Each decoder
block moves another ~16 MB in each direction.

## Why shrinking the caches is safe

Inspection of the graphs shows:

* **FlowLM** writes new keys/values with `ScatterND` at position `step` and reads
  only `Slice[0 : step + len]`. Slots after `step + len` are never read. The
  cache only needs `step + len` slots; only the *declared* dimension forces 1000.
* **Mimi decoder, `encoder_transformer` caches:** in the decoder graph these
  inputs feed nothing but an `Identity` to the matching output. They are
  passthroughs and can be **zero-length**.
* **Mimi decoder, `decoder_transformer` caches** are ring buffers indexed with
  `Mod` on the capacity. Their length is semantically meaningful and **stays
  1000**.

## The fix

1. **Graph:** make the cache axis of those inputs/outputs symbolic
   (`cache_len` / `unused_len`) and drop the stale intermediate shape
   annotations (`value_info`) so onnxruntime re-infers them. Weights and nodes
   are untouched. This runs **in memory at load time**
   ([`src/core/onnx-patch.ts`](../src/core/onnx-patch.ts)), so the stock graphs
   from Hugging Face are used as-is and no preprocessing or re-upload is
   needed. [`tools/make_dynamic_cache.py`](../tools/make_dynamic_cache.py) does
   the same offline with the `onnx` package, if you want prebuilt
   `*_dyncache.onnx` files.
2. **Runtime:** the engine sizes FlowLM caches to the filled prefix plus
   headroom (`cacheCapacityFor`: `step + needed + 32`, rounded up to 64,
   capped at 1000). When a run would overflow, it grows the buffer and copies
   the filled prefix (`resizeCacheData`). Passthrough decoder caches are passed
   empty; ring caches keep 1000 slots.

## Evidence

**Output identity.** The fixed path and the dynamic path produce
**bit-identical PCM** (SHA-256 over the Float32 samples):

| Check | Where | Result |
|---|---|---|
| Stock Alba voice, INT8-wide, 4 WASM threads (Node): "Guten Morgen.", a two-sentence question, a two-sentence paragraph | `npm run test:models` | identical; the cache grew during generation and never needed 1000 slots |
| Stock Alba voice, browser (1 thread), two-sentence text, immediate / queued / complete playback | in-page check, 2026-10-20 | identical PCM in all 6 runs (`8699f78c…`) |
| Embedding from the original project's reference clip, three single-sentence prompts (unaffected by the later per-sentence chunking) | `POCKET_TTS_REFERENCE_F32=… npm run test:models` | identical to hashes from the original `browser-quant-v1` worker (`2be4e041…`, `79d48600…`, `97f41df7…`) |
| Same, four prompts including the 49-word paragraph, before the sentence-ending port | original `bench_worker.mjs --dynamic 1` vs this repo's in-memory patch | identical (`61393e8d…` for the paragraph). Multi-sentence output has since changed by design (one chunk per sentence), so new hashes apply |
| Fake graphs that assert the prefix is intact on every resize, passthrough caches are empty and ring caches are 1000 | `test/engine.test.mjs` | pass |
| Hand-built ONNX protobufs: only the declared dims change; nodes, initializers and other fields are byte-identical; a non-1000 dimension is refused | `test/onnx-patch.test.mjs` | pass |

The identity claim covers the cases above: onnxruntime-web 1.20.0 WASM, the
German bundle at revision `58a6d00`, temperature 0.3, seed 42. Other
backends, versions or languages need their own check (`npm run test:models`
and `bench/bench.mjs --cache both` fail on any mismatch).

**Bytes moved.** These are counted by `bench/bench.mjs` from the actual state
tensors passed into each run. They are exact and independent of timing:

| Prompt | Cache | FlowLM state in, per call | Decoder state in, per call |
|---|---|---|---|
| "Guten Morgen." / 7-word sentence | fixed | 49.15 MB | 16.49 MB |
| same | dynamic | 9.44 MB | 8.30 MB |

The same amount comes back out as outputs. Peak memory was not profiled
separately.

**Speed.** Timing depends on the machine, browser and load. No release
benchmark has been run yet. The protocol is in [`bench/protocol.md`](../bench/protocol.md),
and these earlier numbers are labelled for what they are:

| Source | Setup | Fixed caches | Dynamic caches |
|---|---|---|---|
| Earlier development session (historical, not re-run) | Intel i7-9750H, Node, onnxruntime-web WASM, 4 threads, German INT8-wide | RTF ≈ 1.1 | RTF ≈ 0.6 |
| Earlier development session (historical) | same machine, browser pane, 1 thread | RTF 6.6–10.6 | RTF ≈ 1.2 |
| Earlier development session (historical) | same machine, Chrome, 4 threads, live playback | – | RTF ≈ 0.55–0.65, 0 live gaps |
| 2026-10-20 smoke run, **labelled stress** (load average ≈ 7 on 12 cores) | i7-9750H, Node 22.22.2, onnxruntime-web 1.20.0, 4 threads, INT8-wide, stock Alba, 3-frame delivery, 2 repetitions | RTF 1.78 / 1.63 (2 short prompts), 3–8 live gaps | RTF 0.89 / 0.84, 0 live gaps (≈ 2× faster) |

RTF = wall time ÷ audio duration; below 1 means faster than real time. In the
same session, 6 threads were slower than 4.

## Credits

The dynamic-cache graph patch and runtime allocation were implemented by Claude
(Anthropic) in Ali's development sessions. They were then integrated and
verified alongside Codex work. The runtime is derived from
[vlapky/pocket-tts-js](https://github.com/vlapky/pocket-tts-js). The ONNX
export and INT8 graphs are by
[KevinAHM](https://huggingface.co/KevinAHM/pocket-tts-onnx), and the model is
Kyutai's [Pocket TTS](https://huggingface.co/kyutai/pocket-tts). We have not
surveyed every Pocket TTS port, so we make no claim that this is the first such
change. The upstream export is the natural place for the graph-side change; see
`docs/release-checklist.md`.
