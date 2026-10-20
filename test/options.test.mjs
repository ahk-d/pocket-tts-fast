// Ported from browser-quant-v1/web/runtime/test/runtime-options.test.js.
import test from "node:test";
import assert from "node:assert/strict";
import {
  PRECISION_STEMS,
  modelFilename,
  requiredModelStems,
  requireGermanBos,
  createSeededRandom,
  cacheCapacityFor,
  resizeCacheData,
  GERMAN,
  filesFor,
  downloadBytes,
} from "../dist/index.js";

test("precision profiles quantize the intended graphs while sharing conditioning graphs", () => {
  assert.deepEqual(PRECISION_STEMS.fp32, []);
  assert.deepEqual(PRECISION_STEMS["int8-core"], ["flow_lm_main"]);
  assert.deepEqual(PRECISION_STEMS["int8-wide"], ["flow_lm_main", "flow_lm_flow", "mimi_decoder"]);
  for (const stems of Object.values(PRECISION_STEMS)) {
    assert.equal(stems.includes("mimi_encoder"), false);
    assert.equal(stems.includes("text_conditioner"), false);
  }
  assert.deepEqual(GERMAN.precisions, { fp32: [], "int8-core": ["flow_lm_main"], "int8-wide": ["flow_lm_main", "flow_lm_flow", "mimi_decoder"] });
});

test("selects only explicitly quantized ONNX graphs and never falls back", () => {
  const stems = ["flow_lm_main"];
  assert.equal(modelFilename("flow_lm_main", stems), "flow_lm_main_int8.onnx");
  assert.equal(modelFilename("mimi_decoder", stems), "mimi_decoder.onnx");
  assert.equal(modelFilename("text_conditioner", []), "text_conditioner.onnx");
});

test("prebuilt dynamic-cache graphs are chosen only for the stateful stems", () => {
  assert.equal(modelFilename("flow_lm_main", ["flow_lm_main"], true), "flow_lm_main_int8_dyncache.onnx");
  assert.equal(modelFilename("mimi_decoder", [], true), "mimi_decoder_dyncache.onnx");
  assert.equal(modelFilename("flow_lm_flow", ["flow_lm_flow"], true), "flow_lm_flow_int8.onnx");
  assert.equal(modelFilename("flow_lm_main", ["flow_lm_main"]), "flow_lm_main_int8.onnx");
});

test("fixed-seed noise is repeatable and non-degenerate", () => {
  const a = Array.from({ length: 8 }, createSeededRandom(42));
  const b = Array.from({ length: 8 }, createSeededRandom(42));
  assert.deepEqual(a, b);
  assert.ok(new Set(a).size > 1);
  assert.ok(a.every((v) => v > 0 && v < 1));
});

test("requires German BOS only when the bundle declares insertion", () => {
  const german = { language: "german", insert_bos_before_voice: true };
  assert.throws(() => requireGermanBos(german, null), /German.*BOS/i);
  assert.equal(requireGermanBos(german, { data: new Float32Array([1]), shape: [1, 1, 1] }), true);
  assert.equal(requireGermanBos({ language: "english", insert_bos_before_voice: true }, null), false);
  assert.equal(requireGermanBos({ language: "german", insert_bos_before_voice: false }, null), false);
});

test("stock-voice use excludes the reference encoder", () => {
  assert.deepEqual(requiredModelStems(false), ["text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder"]);
  assert.deepEqual(requiredModelStems(true), ["mimi_encoder", "text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder"]);
  const files = filesFor(GERMAN, "int8-wide").map((a) => a.file);
  assert.equal(files.includes("mimi_encoder.onnx"), false);
  assert.ok(filesFor(GERMAN, "int8-wide", { cloning: true }).some((a) => a.file === "mimi_encoder.onnx"));
});

test("each precision downloads exactly one graph per stem", () => {
  for (const precision of Object.keys(GERMAN.precisions)) {
    const graphs = filesFor(GERMAN, precision).filter((a) => a.file.endsWith(".onnx")).map((a) => a.file.replace(/(_int8)?\.onnx$/, ""));
    assert.deepEqual(graphs.sort(), ["flow_lm_flow", "flow_lm_main", "mimi_decoder", "text_conditioner"]);
  }
  assert.ok(downloadBytes(GERMAN, "int8-wide") < downloadBytes(GERMAN, "int8-core"));
  assert.ok(downloadBytes(GERMAN, "int8-core") < downloadBytes(GERMAN, "fp32"));
  // Graphs + metadata for the default profile, excluding onnxruntime-web itself.
  assert.equal(downloadBytes(GERMAN, "int8-wide"), 24362 + 59837 + 4224 + 16388344 + 76341079 + 9962530 + 22684077);
});

test("every manifest asset carries a SHA-256 and byte size", () => {
  for (const a of [...GERMAN.assets, ...GERMAN.voices]) {
    assert.match(a.sha256, /^[0-9a-f]{64}$/, a.file);
    assert.ok(a.bytes > 0, a.file);
  }
  assert.match(GERMAN.modelBaseUrl, new RegExp(`/resolve/${GERMAN.revision}/`), "graphs come from a commit-pinned URL");
});

test("cache resize keeps the filled prefix of every outer block", () => {
  // shape [2, 3, 2]: two kv halves, three slots, two values per slot
  const data = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const grown = resizeCacheData(data, [2, 3, 2], 1, 2, 4);
  assert.deepEqual([...grown.slice(0, 4)], [1, 2, 3, 4]);
  assert.ok(grown.slice(4, 8).every(Number.isNaN));
  assert.deepEqual([...grown.slice(8, 12)], [7, 8, 9, 10]);
  assert.deepEqual([...resizeCacheData(data, [2, 3, 2], 1, 3, 1)], [1, 2, 7, 8]);
});

test("cache capacity rounds up with headroom and never exceeds the graph maximum", () => {
  assert.equal(cacheCapacityFor(126), 192);
  assert.equal(cacheCapacityFor(128, 0), 128);
  assert.equal(cacheCapacityFor(990), 1000);
  assert.equal(cacheCapacityFor(1000), 1000);
});
