// In-memory version of tools/make_dynamic_cache.py.
//
// Rewrites the declared shape of selected graph inputs/outputs in an ONNX
// ModelProto from a fixed 1000 to a symbolic dimension, and drops the graph's
// intermediate shape annotations (value_info) so onnxruntime re-infers them.
// Weights and nodes are copied byte-for-byte; nothing else changes. This lets
// the runtime load the stock graphs from Hugging Face and still size the KV
// caches itself, without a preprocessing step.
//
// Only the protobuf fields involved are decoded:
//   ModelProto.graph = 7
//   GraphProto.node = 1, input = 11, output = 12, value_info = 13
//   NodeProto.input = 1, op_type = 4
//   ValueInfoProto.name = 1, type = 2
//   TypeProto.tensor_type = 1; TypeProto.Tensor.shape = 2
//   TensorShapeProto.dim = 1; Dimension.dim_value = 1, dim_param = 2

import { utf8Decode, utf8Encode } from "./utf8.js";

interface Field {
  field: number;
  wire: number;
  /** Offset of the tag. */
  start: number;
  /** Offset of the payload (after tag and, for wire 2, the length). */
  valueStart: number;
  /** End of the payload. */
  end: number;
}

function readVarint(bytes: Uint8Array, pos: number): [number, number] {
  let result = 0;
  let mul = 1;
  let i = pos;
  for (;;) {
    if (i >= bytes.length) throw new Error("Truncated protobuf varint");
    const b = bytes[i++];
    result += (b & 0x7f) * mul;
    if ((b & 0x80) === 0) break;
    mul *= 128;
  }
  return [result, i];
}

function encodeVarint(value: number): Uint8Array {
  const out: number[] = [];
  let v = value;
  while (v >= 0x80) {
    out.push((v % 128) | 0x80);
    v = Math.floor(v / 128);
  }
  out.push(v);
  return Uint8Array.from(out);
}

function fields(bytes: Uint8Array, start: number, end: number): Field[] {
  const out: Field[] = [];
  let i = start;
  while (i < end) {
    const fieldStart = i;
    let tag: number;
    [tag, i] = readVarint(bytes, i);
    const field = Math.floor(tag / 8);
    const wire = tag & 7;
    let valueStart = i;
    if (wire === 0) [, i] = readVarint(bytes, i);
    else if (wire === 1) i += 8;
    else if (wire === 5) i += 4;
    else if (wire === 2) {
      let len: number;
      [len, valueStart] = readVarint(bytes, i);
      i = valueStart + len;
    } else throw new Error(`Unsupported protobuf wire type ${wire}`);
    if (i > end) throw new Error("Truncated protobuf message");
    out.push({ field, wire, start: fieldStart, valueStart, end: i });
  }
  return out;
}

function stringField(bytes: Uint8Array, f: Field): string {
  return utf8Decode(bytes.subarray(f.valueStart, f.end));
}

type Part = Uint8Array;

function lengthDelimited(field: number, parts: Part[]): Part[] {
  const len = parts.reduce((n, p) => n + p.length, 0);
  return [encodeVarint(field * 8 + 2), encodeVarint(len), ...parts];
}

function concat(parts: Part[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

/** Dimensions of a ValueInfoProto tensor type: fixed value, symbolic name or unknown. */
export function valueInfoDims(bytes: Uint8Array, start: number, end: number): (number | string | null)[] {
  const dims: (number | string | null)[] = [];
  for (const t of fields(bytes, start, end)) {
    if (t.field !== 2 || t.wire !== 2) continue;
    for (const tt of fields(bytes, t.valueStart, t.end)) {
      if (tt.field !== 1 || tt.wire !== 2) continue;
      for (const s of fields(bytes, tt.valueStart, tt.end)) {
        if (s.field !== 2 || s.wire !== 2) continue;
        for (const d of fields(bytes, s.valueStart, s.end)) {
          if (d.field !== 1 || d.wire !== 2) continue;
          let dim: number | string | null = null;
          for (const x of fields(bytes, d.valueStart, d.end)) {
            if (x.field === 1 && x.wire === 0) dim = readVarint(bytes, x.valueStart)[0];
            else if (x.field === 2 && x.wire === 2) dim = stringField(bytes, x);
          }
          dims.push(dim);
        }
      }
    }
  }
  return dims;
}

// Rebuild a ValueInfoProto with dimension `axis` replaced by `dimParam`.
function rewriteValueInfo(bytes: Uint8Array, vi: Field, axis: number, dimParam: string, expected: number): Part[] {
  const rebuild = (start: number, end: number, onField: (f: Field) => Part[] | null): Part[] => {
    const parts: Part[] = [];
    for (const f of fields(bytes, start, end)) {
      const replaced = onField(f);
      parts.push(...(replaced ?? [bytes.subarray(f.start, f.end)]));
    }
    return parts;
  };
  let seen = false;
  const dimension = (d: Field): Part[] => {
    let value: number | null = null;
    const keep: Part[] = [];
    for (const x of fields(bytes, d.valueStart, d.end)) {
      if (x.field === 1 && x.wire === 0) value = readVarint(bytes, x.valueStart)[0];
      else if (x.field !== 2) keep.push(bytes.subarray(x.start, x.end)); // denotation
    }
    if (value !== expected) throw new Error(`Expected dimension ${expected} at axis ${axis}, found ${value}`);
    seen = true;
    const name = utf8Encode(dimParam);
    return lengthDelimited(1, [encodeVarint(2 * 8 + 2), encodeVarint(name.length), name, ...keep]);
  };
  const shape = (s: Field): Part[] => {
    let index = 0;
    return lengthDelimited(
      2,
      rebuild(s.valueStart, s.end, (d) => (d.field === 1 && d.wire === 2 && index++ === axis ? dimension(d) : null)),
    );
  };
  const tensor = (tt: Field): Part[] =>
    lengthDelimited(1, rebuild(tt.valueStart, tt.end, (s) => (s.field === 2 && s.wire === 2 ? shape(s) : null)));
  const type = (t: Field): Part[] =>
    lengthDelimited(2, rebuild(t.valueStart, t.end, (tt) => (tt.field === 1 && tt.wire === 2 ? tensor(tt) : null)));
  const parts = lengthDelimited(vi.field, rebuild(vi.valueStart, vi.end, (t) => (t.field === 2 && t.wire === 2 ? type(t) : null)));
  if (!seen) throw new Error(`Axis ${axis} not found`);
  return parts;
}

export interface DynamicAxisRule {
  /** Graph input to make dynamic. */
  inputName: string;
  /** Matching graph output (the updated state). */
  outputName: string;
  axis: number;
  dimParam: string;
  /** Only apply if the input's sole consumer is an Identity node (passthrough cache). */
  identityOnly?: boolean;
}

export interface PatchResult {
  bytes: Uint8Array;
  /** Inputs whose cache axis is now symbolic. */
  changed: string[];
}

/**
 * Make the cache axis of the selected inputs/outputs symbolic. Throws if a rule
 * targets a dimension that is not `expectedDim` (the stock 1000 slots).
 */
export function makeCacheDynamic(model: Uint8Array, rules: readonly DynamicAxisRule[], expectedDim = 1000): PatchResult {
  const top = fields(model, 0, model.length);
  const graphField = top.find((f) => f.field === 7 && f.wire === 2);
  if (!graphField) throw new Error("ONNX model has no graph");
  const graph = fields(model, graphField.valueStart, graphField.end);

  const nameOf = (vi: Field) => {
    for (const f of fields(model, vi.valueStart, vi.end)) if (f.field === 1 && f.wire === 2) return stringField(model, f);
    return "";
  };

  let consumers: Map<string, string[]> | null = null;
  if (rules.some((r) => r.identityOnly)) {
    consumers = new Map();
    for (const node of graph) {
      if (node.field !== 1 || node.wire !== 2) continue;
      const inputs: string[] = [];
      let op = "";
      for (const f of fields(model, node.valueStart, node.end)) {
        if (f.field === 1 && f.wire === 2) inputs.push(stringField(model, f));
        else if (f.field === 4 && f.wire === 2) op = stringField(model, f);
      }
      for (const name of inputs) {
        const list = consumers.get(name) ?? [];
        list.push(op);
        consumers.set(name, list);
      }
    }
  }

  const byInput = new Map<string, DynamicAxisRule>();
  for (const rule of rules) {
    if (rule.identityOnly) {
      const ops = consumers!.get(rule.inputName) ?? [];
      if (!(ops.length === 1 && ops[0] === "Identity")) continue;
    }
    byInput.set(rule.inputName, rule);
  }
  const byOutput = new Map([...byInput.values()].map((r) => [r.outputName, r]));

  const changed: string[] = [];
  const patchedOutputs = new Set<string>();
  const graphParts: Part[] = [];
  for (const f of graph) {
    if (f.field === 13) continue; // value_info: stale shapes, let the runtime re-infer
    if ((f.field === 11 || f.field === 12) && f.wire === 2) {
      const name = nameOf(f);
      const rule = f.field === 11 ? byInput.get(name) : byOutput.get(name);
      if (rule) {
        graphParts.push(...rewriteValueInfo(model, f, rule.axis, rule.dimParam, expectedDim));
        if (f.field === 11) changed.push(name);
        else patchedOutputs.add(name);
        continue;
      }
    }
    graphParts.push(model.subarray(f.start, f.end));
  }
  for (const rule of byInput.values()) {
    if (!changed.includes(rule.inputName)) throw new Error(`Graph input ${rule.inputName} not found`);
    if (!patchedOutputs.has(rule.outputName)) throw new Error(`Graph output ${rule.outputName} not found`);
  }

  const parts: Part[] = [];
  for (const f of top) {
    if (f === graphField) parts.push(...lengthDelimited(7, graphParts));
    else parts.push(model.subarray(f.start, f.end));
  }
  return { bytes: concat(parts), changed };
}

/** Rule set for the graph inputs of `flow_lm_main` / `mimi_decoder`, from bundle.json. */
export function dynamicCacheRules(
  stem: "flow_lm_main" | "mimi_decoder",
  bundle: {
    flow_lm_state_manifest: readonly { input_name: string; output_name: string; key: string }[];
    mimi_state_manifest: readonly { input_name: string; output_name: string; key: string }[];
  },
): DynamicAxisRule[] {
  if (stem === "flow_lm_main") {
    return bundle.flow_lm_state_manifest
      .filter((e) => e.key === "cache")
      .map((e) => ({ inputName: e.input_name, outputName: e.output_name, axis: 2, dimParam: "cache_len" }));
  }
  // Only the decoder's passthrough (Identity) caches; its ring caches use Mod on capacity.
  return bundle.mimi_state_manifest
    .filter((e) => e.key === "cache")
    .map((e) => ({ inputName: e.input_name, outputName: e.output_name, axis: 3, dimParam: "unused_len", identityOnly: true }));
}
