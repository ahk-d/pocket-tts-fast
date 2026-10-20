// The in-memory dynamic-cache patch, on small hand-built ONNX ModelProtos.
import test from "node:test";
import assert from "node:assert/strict";
import { makeCacheDynamic, dynamicCacheRules, valueInfoDims } from "../dist/index.js";

// --- minimal protobuf writer/reader for the test ---
const varint = (n) => {
  const out = [];
  while (n >= 0x80) {
    out.push((n % 128) | 0x80);
    n = Math.floor(n / 128);
  }
  out.push(n);
  return out;
};
const cat = (...parts) => Uint8Array.from(parts.flatMap((p) => [...p]));
const ld = (field, payload) => cat(varint(field * 8 + 2), varint(payload.length), payload);
const vint = (field, n) => cat(varint(field * 8), varint(n));
const str = (field, s) => ld(field, new TextEncoder().encode(s));
const dim = (v) => (typeof v === "number" ? ld(1, vint(1, v)) : ld(1, str(2, v)));
const valueInfo = (field, name, dims) => ld(field, cat(str(1, name), ld(2, ld(1, cat(vint(1, 1), ld(2, cat(...dims.map(dim))))))));
const node = (inputs, op) => ld(1, cat(...inputs.map((i) => str(1, i)), str(4, op)));

function readFields(bytes, start = 0, end = bytes.length) {
  const out = [];
  let i = start;
  const rv = () => {
    let r = 0, m = 1, b;
    do {
      b = bytes[i++];
      r += (b & 127) * m;
      m *= 128;
    } while (b & 128);
    return r;
  };
  while (i < end) {
    const tag = rv();
    const f = { field: Math.floor(tag / 8), wire: tag & 7 };
    if (f.wire === 0) rv();
    else if (f.wire === 2) {
      const len = rv();
      f.valueStart = i;
      i += len;
      f.end = i;
    }
    out.push(f);
  }
  return out;
}
function graphOf(model) {
  const g = readFields(model).find((f) => f.field === 7);
  return readFields(model, g.valueStart, g.end).map((f) => ({ ...f, bytes: model.subarray(f.valueStart, f.end) }));
}
function nameOf(model, f) {
  const n = readFields(model, f.valueStart, f.end).find((x) => x.field === 1);
  return new TextDecoder().decode(model.subarray(n.valueStart, n.end));
}
function dimsByName(model, field) {
  const out = {};
  for (const f of graphOf(model).filter((x) => x.field === field)) out[nameOf(model, f)] = valueInfoDims(model, f.valueStart, f.end);
  return out;
}

const weights = Uint8Array.from({ length: 300 }, (_, i) => (i * 37) % 256); // > 127 bytes: multi-byte lengths
const flowModel = cat(
  vint(1, 8),
  ld(7, cat(
    node(["state_0", "x"], "Slice"),
    ld(5, cat(str(1, "w"), ld(9, weights))),
    valueInfo(11, "state_0", [2, 1, 1000, 16, 64]),
    valueInfo(11, "state_1", [1]),
    valueInfo(11, "x", ["batch", 32]),
    valueInfo(12, "out_state_0", [2, 1, 1000, 16, 64]),
    valueInfo(13, "stale", [2, 1, 1000, 16, 64]),
  )),
  ld(8, cat(str(1, ""), vint(2, 17))),
);
const bundle = {
  flow_lm_state_manifest: [
    { input_name: "state_0", output_name: "out_state_0", key: "cache" },
    { input_name: "state_1", output_name: "out_state_1", key: "step" },
  ],
  mimi_state_manifest: [
    { input_name: "enc_cache", output_name: "out_enc_cache", key: "cache" },
    { input_name: "ring_cache", output_name: "out_ring_cache", key: "cache" },
  ],
};

test("FlowLM cache axis becomes symbolic; weights, nodes and other inputs are byte-identical", () => {
  const { bytes, changed } = makeCacheDynamic(flowModel, dynamicCacheRules("flow_lm_main", bundle));
  assert.deepEqual(changed, ["state_0"]);
  assert.deepEqual(dimsByName(bytes, 11), { state_0: [2, 1, "cache_len", 16, 64], state_1: [1], x: ["batch", 32] });
  assert.deepEqual(dimsByName(bytes, 12), { out_state_0: [2, 1, "cache_len", 16, 64] });
  assert.equal(graphOf(bytes).filter((f) => f.field === 13).length, 0, "stale value_info dropped");
  const before = graphOf(flowModel).filter((f) => f.field === 1 || f.field === 5);
  const after = graphOf(bytes).filter((f) => f.field === 1 || f.field === 5);
  assert.deepEqual(after.map((f) => [...f.bytes]), before.map((f) => [...f.bytes]), "nodes and initializers untouched");
  // Fields outside the graph (opset_import) are preserved in order.
  assert.deepEqual(readFields(bytes).map((f) => f.field), [1, 7, 8]);
});

test("refuses a cache dimension that is not the stock 1000 slots (e.g. patching twice)", () => {
  const { bytes } = makeCacheDynamic(flowModel, dynamicCacheRules("flow_lm_main", bundle));
  assert.throws(() => makeCacheDynamic(bytes, dynamicCacheRules("flow_lm_main", bundle)), /Expected dimension 1000/);
});

test("decoder: only Identity-passthrough caches change; Mod ring caches keep 1000 slots", () => {
  const decoder = cat(
    ld(7, cat(
      node(["enc_cache"], "Identity"),
      node(["ring_cache", "pos"], "Mod"),
      valueInfo(11, "enc_cache", [2, 1, 8, 1000, 64]),
      valueInfo(11, "ring_cache", [2, 1, 8, 1000, 64]),
      valueInfo(12, "out_enc_cache", [2, 1, 8, 1000, 64]),
      valueInfo(12, "out_ring_cache", [2, 1, 8, 1000, 64]),
    )),
  );
  const { bytes, changed } = makeCacheDynamic(decoder, dynamicCacheRules("mimi_decoder", bundle));
  assert.deepEqual(changed, ["enc_cache"]);
  assert.deepEqual(dimsByName(bytes, 11), { enc_cache: [2, 1, 8, "unused_len", 64], ring_cache: [2, 1, 8, 1000, 64] });
  assert.deepEqual(dimsByName(bytes, 12), { out_enc_cache: [2, 1, 8, "unused_len", 64], out_ring_cache: [2, 1, 8, 1000, 64] });
});

test("missing graph output for a rule is an error, not a silent partial patch", () => {
  const broken = cat(ld(7, cat(valueInfo(11, "state_0", [2, 1, 1000, 16, 64]))));
  assert.throws(() => makeCacheDynamic(broken, dynamicCacheRules("flow_lm_main", bundle)), /output out_state_0 not found/);
  assert.throws(() => makeCacheDynamic(cat(vint(1, 8)), []), /no graph/);
});
