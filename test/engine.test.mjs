// The generation loop against fake sessions that check what the real graphs
// rely on: every FlowLM run gets a cache with room for step + len slots whose
// filled prefix is intact, passthrough caches are empty, ring caches stay 1000.
import test from "node:test";
import assert from "node:assert/strict";
import { PocketTTSEngine, encodeNpyFloat32, validateEmbedding, cacheCapacityFor } from "../dist/index.js";

const LATENT = 2;
const COND = 4;
const SLOTS = 1000;

// SentencePiece ModelProto with one piece per printable ASCII char plus "▁".
function tokenizerModel() {
  const enc = new TextEncoder();
  const bytes = [];
  const varint = (n) => {
    while (n >= 0x80) {
      bytes.push((n & 127) | 128);
      n >>>= 7;
    }
    bytes.push(n);
  };
  const chars = ["▁", ...Array.from({ length: 95 }, (_, i) => String.fromCharCode(32 + i)), "ä", "ü"];
  for (const ch of chars) {
    const p = enc.encode(ch);
    const piece = [0x0a, p.length, ...p, 0x15, 0, 0, 0x80, 0xbf]; // score -1.0f
    bytes.push(0x0a);
    varint(piece.length);
    bytes.push(...piece);
  }
  return Uint8Array.from(bytes);
}

const bundle = {
  language: "german",
  sample_rate: 100,
  samples_per_frame: 10,
  latent_dim: LATENT,
  conditioning_dim: COND,
  max_token_per_chunk: 30,
  tokenizer_file: "tokenizer.model",
  bos_before_voice_file: "bos.npy",
  insert_bos_before_voice: true,
  remove_semicolons: true,
  flow_lm_state_manifest: [
    { input_name: "state_0", output_name: "out_state_0", module: "transformer.layers.0.self_attn", key: "cache", shape: [2, 1, SLOTS, 2, 2], dtype: "float32", fill: "nan" },
    { input_name: "state_1", output_name: "out_state_1", module: "transformer.layers.0.self_attn", key: "step", shape: [1], dtype: "int64", fill: "zeros" },
  ],
  mimi_state_manifest: [
    { input_name: "m_0", output_name: "out_m_0", module: "encoder_transformer.layers.0.self_attn", key: "cache", shape: [2, 1, 2, SLOTS, 2], dtype: "float32", fill: "zeros" },
    { input_name: "m_1", output_name: "out_m_1", module: "decoder_transformer.layers.0.self_attn", key: "cache", shape: [2, 1, 2, SLOTS, 2], dtype: "float32", fill: "zeros" },
    { input_name: "m_2", output_name: "out_m_2", module: "decoder.model.0", key: "first", shape: [1], dtype: "bool", fill: "ones" },
  ],
};

function fakeHost({ framesPerChunk = 6, bundleOverrides = {} } = {}) {
  const log = { loaded: [], flowSlots: [], prefixErrors: 0, encCache: [], ringCache: [], decodeSizes: [], encoderRuns: 0 };
  const tensor = (type, data, dims) => ({ type, data, dims: [...dims] });
  let framesInChunk = 0;
  const sessions = {
    text_conditioner: {
      outputNames: ["emb"],
      run: async ({ token_ids }) => ({ emb: tensor("float32", new Float32Array(token_ids.dims[1] * COND).fill(0.5), [1, token_ids.dims[1], COND]) }),
    },
    flow_lm_main: {
      outputNames: ["conditioning", "eos_logit", "out_state_0", "out_state_1"],
      run: async (feeds) => {
        const cache = feeds.state_0;
        const step = Number(feeds.state_1.data[0]);
        const len = feeds.sequence.dims[1] + feeds.text_embeddings.dims[1];
        const slots = cache.dims[2];
        log.flowSlots.push(slots);
        if (slots < step + len) throw new Error(`cache too small: ${slots} < ${step + len}`);
        // Slot s of every outer block holds the marker s+1 once written.
        const inner = 4;
        const data = Float32Array.from(cache.data);
        for (let o = 0; o < 2; o++) {
          for (let s = 0; s < step; s++) if (data[(o * slots + s) * inner] !== s + 1) log.prefixErrors++;
          for (let s = step; s < step + len; s++) data.fill(s + 1, (o * slots + s) * inner, (o * slots + s + 1) * inner);
        }
        if (feeds.text_embeddings.dims[1] > 0) framesInChunk = 0;
        else framesInChunk++;
        return {
          conditioning: tensor("float32", new Float32Array(COND), [1, 1, COND]),
          eos_logit: tensor("float32", new Float32Array([framesInChunk >= framesPerChunk ? 0 : -10]), [1, 1]),
          out_state_0: tensor("float32", data, cache.dims),
          out_state_1: tensor("int64", BigInt64Array.from([BigInt(step + len)]), [1]),
        };
      },
    },
    flow_lm_flow: { outputNames: ["flow_dir"], run: async () => ({ flow_dir: tensor("float32", new Float32Array(LATENT), [1, LATENT]) }) },
    mimi_decoder: {
      outputNames: ["audio", "out_m_0", "out_m_1", "out_m_2"],
      run: async (feeds) => {
        const n = feeds.latent.dims[1];
        log.decodeSizes.push(n);
        log.encCache.push(feeds.m_0.dims[3]);
        log.ringCache.push(feeds.m_1.dims[3]);
        return { audio: tensor("float32", new Float32Array(n * 10).fill(0.1), [1, 1, n * 10]), out_m_0: feeds.m_0, out_m_1: feeds.m_1, out_m_2: feeds.m_2 };
      },
    },
    mimi_encoder: {
      outputNames: ["emb"],
      run: async ({ audio }) => {
        log.encoderRuns++;
        const frames = Math.max(1, Math.floor(audio.dims[2] / 10));
        return { emb: tensor("float32", new Float32Array(frames * COND).fill(0.25), [1, 1, frames, COND]) };
      },
    },
  };
  const files = {
    "bundle.json": new TextEncoder().encode(JSON.stringify({ ...bundle, ...bundleOverrides })),
    "tokenizer.model": tokenizerModel(),
    "bos.npy": encodeNpyFloat32(new Float32Array(COND).fill(1), [1, 1, COND]),
    "voices/v.npy": encodeNpyFloat32(new Float32Array(5 * COND).fill(0.3), [1, 5, COND]),
  };
  return {
    log,
    host: {
      tensor,
      readAsset: async (file) => {
        if (!files[file]) throw new Error(`404 ${file}`);
        return files[file];
      },
      loadModel: async ({ stem, filename, transform }) => {
        log.loaded.push({ stem, filename, patched: Boolean(transform) });
        return sessions[stem];
      },
    },
  };
}

async function engineWith(opts = {}, hostOpts) {
  const { host, log } = fakeHost(hostOpts);
  const engine = await PocketTTSEngine.load(host, { yieldToEventLoop: async () => {}, ...opts });
  return { engine, log };
}

test("stock use loads no encoder; graphs that hold caches are patched in memory", async () => {
  const { engine, log } = await engineWith();
  assert.deepEqual(log.loaded.map((l) => l.stem), ["text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder"]);
  assert.deepEqual(log.loaded.filter((l) => l.patched).map((l) => l.stem), ["flow_lm_main", "mimi_decoder"]);
  assert.deepEqual(log.loaded.map((l) => l.filename), ["text_conditioner.onnx", "flow_lm_main_int8.onnx", "flow_lm_flow_int8.onnx", "mimi_decoder_int8.onnx"]);
  await engine.loadEmbeddingAsset("voices/v.npy", "stock:v");
  assert.equal(log.loaded.length, 4);
});

test("dynamic caches: sized to the context, grown with the prefix intact, passthrough empty, ring caches 1000", async () => {
  const { engine, log } = await engineWith({}, { framesPerChunk: 200 });
  const ref = await engine.loadEmbeddingAsset("voices/v.npy", "stock:v");
  const metrics = await engine.generate("Hallo Welt.", ref, { deliveryFrames: 12 });
  assert.equal(log.prefixErrors, 0, "filled prefix preserved through every resize");
  // Voice conditioning: bos (1) + embedding (5) = 6 slots exactly.
  assert.equal(log.flowSlots[0], 6);
  const genSlots = log.flowSlots.slice(1);
  assert.ok(genSlots.every((s) => s < SLOTS), "never the full 1000 slots");
  assert.equal(genSlots[0], cacheCapacityFor(6 + engine.tokenizer.encodeIds("Hallo Welt.").length));
  assert.ok(new Set(genSlots).size > 1, "the cache grew during the utterance");
  assert.ok(log.encCache.every((d) => d === 0), "encoder_transformer passthrough caches are empty");
  assert.ok(log.ringCache.every((d) => d === SLOTS), "decoder ring caches keep 1000 slots");
  // EOS at frame 200, then 3 more frames for a short (<= 4 word) text.
  assert.equal(metrics.frames, 203);
});

test("fixed caches (dynamicCache: false) pass 1000 slots everywhere and produce the same audio", async () => {
  const fixed = await engineWith({ dynamicCache: false }, { framesPerChunk: 30 });
  const dyn = await engineWith({}, { framesPerChunk: 30 });
  assert.ok(fixed.log.loaded.every((l) => !l.patched));
  const run = async ({ engine }) => {
    const ref = await engine.loadEmbeddingAsset("voices/v.npy", "v");
    const chunks = [];
    await engine.generate("Guten Morgen.", ref, { deliveryFrames: 3, onChunk: (a) => chunks.push(...a) });
    return chunks;
  };
  const a = await run(fixed);
  const b = await run(dyn);
  assert.ok(fixed.log.flowSlots.every((s) => s === SLOTS));
  assert.ok(fixed.log.encCache.every((d) => d === SLOTS));
  assert.equal(fixed.log.prefixErrors, 0);
  assert.deepEqual(a, b);
});

test("delivery: a 3-frame first block, then deliveryFrames, then the remainder; pause between text chunks", async () => {
  const { engine, log } = await engineWith({}, { framesPerChunk: 20 });
  const ref = await engine.loadEmbeddingAsset("voices/v.npy", "v");
  const metas = [];
  // Two sentences that don't fit one 30-token chunk (this tokenizer: ~1 token per char).
  const long = "Erster Satz hier. Zweiter Satz da.";
  await engine.generate(long, ref, { deliveryFrames: 12, onChunk: (a, m) => metas.push({ n: a.length, ...m }) });
  // Each sentence is its own chunk with a 3-frame EOS tail (3 words each):
  // EOS at frame 20 + 3 = 23 frames -> 3 + 12 + 8, then 12 + 11
  // (only the very first block of the utterance is 3 frames).
  assert.deepEqual(log.decodeSizes, [3, 12, 8, 12, 11]);
  const silence = metas.filter((m) => m.isSilence);
  assert.equal(silence.length, 1);
  assert.equal(silence[0].n, 25, "0.25 s pause at 100 Hz");
  assert.equal(metas.filter((m) => m.isFirst).length, 1);
  assert.equal(metas.at(-1).isLast, true);
  assert.equal(metas.filter((m) => m.chunkStart).length, 2);
});

test("stop() ends generation after the current frame; maxFrames caps warm-ups", async () => {
  const { engine } = await engineWith({}, { framesPerChunk: 400 });
  const ref = await engine.loadEmbeddingAsset("voices/v.npy", "v");
  let blocks = 0;
  const m = await engine.generate("Hallo.", ref, {
    onChunk: () => {
      if (++blocks === 2) engine.stop();
    },
  });
  assert.equal(m.stopped, true);
  assert.ok(m.frames <= 16);
  const capped = await engine.generate("Hallo.", ref, { maxFrames: 5 });
  assert.equal(capped.frames, 5);
  assert.equal(capped.stopped, true);
  await assert.rejects(engine.generate("   ", ref), /No text/);
  await assert.rejects(engine.generate("Hallo.", "nope"), /not prepared/);
  await assert.rejects(engine.generate("Hallo.", ref, { deliveryFrames: 0 }), /deliveryFrames/);
});

test("overlapping generate() calls are rejected", async () => {
  const { engine } = await engineWith({}, { framesPerChunk: 50 });
  const ref = await engine.loadEmbeddingAsset("voices/v.npy", "v");
  const first = engine.generate("Hallo.", ref);
  await assert.rejects(engine.generate("Hallo.", ref), /already in progress/);
  await first;
});

test("the encoder loads lazily, once, even for concurrent encodes; embeddings are validated", async () => {
  const { engine, log } = await engineWith();
  const audio = new Float32Array(240).fill(0.1);
  const [a, b] = await Promise.all([engine.encodeVoice(audio), engine.encodeVoice(audio)]);
  assert.equal(log.loaded.filter((l) => l.stem === "mimi_encoder").length, 1);
  assert.deepEqual(a.shape, [1, 24, COND]);
  assert.deepEqual(a.data, b.data);
  const ref = await engine.loadVoiceEmbedding(a, "clone:1");
  assert.ok(engine.hasVoice(ref));
  assert.throws(() => validateEmbedding({ data: new Float32Array(8), shape: [1, 2, 3] }, COND), /shape/);
  assert.throws(() => validateEmbedding({ data: Float32Array.from([NaN, 0, 0, 0]), shape: [1, 1, 4] }, COND), /non-finite/);
  await assert.rejects(engine.loadVoiceEmbedding({ data: new Float32Array(3), shape: [1, 1, 3] }, "bad"), /does not match/);
  assert.equal(engine.unloadVoice(ref), true);
});

test("a German bundle without its BOS asset fails to load", async () => {
  const { host } = fakeHost();
  const read = host.readAsset;
  host.readAsset = async (f) => (f === "bos.npy" ? Promise.reject(new Error("404")) : read(f));
  await assert.rejects(PocketTTSEngine.load(host, {}), /BOS/);
});

// Ported from codex-pocket-runtime-2026-10-20/runtime/test/chunk-diagnostics.test.js
// (there: the worker source run in a VM with mocked ONNX; here: the engine with fake sessions).
test("generation reports each text chunk with its conditioning and delivery timing", async () => {
  const { engine } = await engineWith({}, { framesPerChunk: 1 });
  const ref = await engine.loadEmbeddingAsset("voices/v.npy", "v");
  const metas = [];
  const m = await engine.generate("Hello there. How are you?", ref, { deliveryFrames: 3, onChunk: (_, meta) => metas.push(meta) });
  assert.equal(m.textChunks.length, 2);
  assert.deepEqual(m.textChunks.map((c) => c.text), ["Hello there.", "How are you?"]);
  for (const c of m.textChunks) {
    assert.ok(c.conditioningSeconds >= c.startSeconds);
    assert.ok(c.startSeconds <= c.firstAudioSeconds);
    assert.ok(c.firstAudioSeconds <= c.completionSeconds);
    // EOS on the first frame, then a 3-frame tail for <= 4 words: 4 frames, all decoded.
    assert.equal(c.generatedFrames, 4);
    assert.equal(c.emittedFrames, 4);
    assert.equal(c.framesAfterEos, 3);
    assert.equal(c.endReason, "eos");
  }
  assert.equal(m.complete, true);
  assert.equal(metas.filter((x) => !x.isSilence).length, 4, "3 + 1 frames per sentence with 3-frame delivery");
  assert.equal(metas.filter((x) => x.isSilence).length, 1);
});

test("a short sentence in a longer prompt gets the standalone EOS tail", async () => {
  const { engine } = await engineWith({}, { framesPerChunk: 1 });
  const ref = await engine.loadEmbeddingAsset("voices/v.npy", "v");
  const m = await engine.generate("Hello! We go home now ok. Are you coming?", ref);
  assert.equal(m.textChunks.length, 3);
  assert.deepEqual(m.textChunks.map((c) => c.framesAfterEos), [3, 1, 3]);
  assert.deepEqual(m.textChunks.map((c) => c.emittedFrames), [4, 2, 4]);
  assert.ok(m.textChunks.every((c) => c.endReason === "eos"));
});

test("a model EOS-tail recommendation still overrides the short-sentence heuristic", async () => {
  const { engine } = await engineWith({}, { framesPerChunk: 1, bundleOverrides: { model_recommended_frames_after_eos: 2 } });
  const ref = await engine.loadEmbeddingAsset("voices/v.npy", "v");
  const m = await engine.generate("Hello! We go home now ok. Are you coming?", ref);
  assert.deepEqual(m.textChunks.map((c) => c.emittedFrames), [3, 3, 3]);
});

test("the final block is decoded and delivered before a chunk ends (no dropped tail)", async () => {
  const { engine } = await engineWith({}, { framesPerChunk: 5 });
  const ref = await engine.loadEmbeddingAsset("voices/v.npy", "v");
  let samples = 0;
  const m = await engine.generate("Guten Morgen.", ref, { deliveryFrames: 12, onChunk: (a) => (samples += a.length) });
  // EOS at frame 5, 3-frame tail -> 8 frames: 3-frame first block + the 5 pending ones.
  assert.equal(m.frames, 8);
  assert.equal(samples, 8 * 10, "every generated frame reaches the listener");
});

test("generation reports a frame-budget stop without calling it EOS", async () => {
  const { engine } = await engineWith({}, { framesPerChunk: 1 });
  const ref = await engine.loadEmbeddingAsset("voices/v.npy", "v");
  const m = await engine.generate("Hello there. How are you?", ref, { maxFrames: 1 });
  assert.equal(m.textChunks.length, 1);
  assert.equal(m.textChunks[0].endReason, "frame-budget");
  assert.equal(m.textChunks[0].generatedFrames, 1);
  assert.equal(m.textChunks[0].emittedFrames, 0);
  assert.equal(m.textChunks[0].firstAudioSeconds, null);
  assert.equal(m.complete, false);
});

test("a cancelled chunk is reported as cancelled, not complete", async () => {
  const { engine } = await engineWith({}, { framesPerChunk: 100 });
  const ref = await engine.loadEmbeddingAsset("voices/v.npy", "v");
  const m = await engine.generate("Hallo.", ref, { onChunk: () => engine.stop() });
  assert.equal(m.textChunks[0].endReason, "cancelled");
  assert.equal(m.complete, false);
});

test("a chunk that never signals EOS stops at the 500-frame cap and still delivers its audio", async () => {
  const { engine } = await engineWith({}, { framesPerChunk: 10_000 });
  const ref = await engine.loadEmbeddingAsset("voices/v.npy", "v");
  let samples = 0;
  const m = await engine.generate("Hallo.", ref, { onChunk: (a) => (samples += a.length) });
  assert.equal(m.textChunks[0].endReason, "frame-cap");
  assert.equal(m.textChunks[0].generatedFrames, 500);
  assert.equal(m.textChunks[0].emittedFrames, 500);
  assert.equal(samples, 500 * 10);
  assert.equal(m.complete, false, "capped output is not a complete generation");
});
