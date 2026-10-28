export type ModelStem = "mimi_encoder" | "text_conditioner" | "flow_lm_main" | "flow_lm_flow" | "mimi_decoder";

export const MODEL_STEMS: readonly ModelStem[] = Object.freeze([
  "mimi_encoder",
  "text_conditioner",
  "flow_lm_main",
  "flow_lm_flow",
  "mimi_decoder",
]);

/**
 * INT8 graph sets per precision profile. All keep `mimi_encoder` and
 * `text_conditioner` in FP32, so a voice embedding and text conditioning are
 * shared across variants.
 */
export const PRECISION_STEMS = Object.freeze({
  fp32: Object.freeze([] as ModelStem[]),
  "int8-core": Object.freeze(["flow_lm_main"] as ModelStem[]),
  "int8-wide": Object.freeze(["flow_lm_main", "flow_lm_flow", "mimi_decoder"] as ModelStem[]),
});

export type PrecisionName = keyof typeof PRECISION_STEMS;

/** Graphs whose KV caches can be sized by the caller (see make_dynamic_cache.py). */
export const DYNAMIC_CACHE_STEMS: readonly ModelStem[] = Object.freeze(["flow_lm_main", "mimi_decoder"]);

/**
 * File name of a graph. With `prebuiltDynamic`, the stateful stems resolve to
 * the `*_dyncache.onnx` copies written by tools/make_dynamic_cache.py.
 */
export function modelFilename(name: ModelStem, quantizedStems: readonly string[] = [], prebuiltDynamic = false): string {
  const int8 = quantizedStems.includes(name) ? "_int8" : "";
  const dyn = prebuiltDynamic && DYNAMIC_CACHE_STEMS.includes(name) ? "_dyncache" : "";
  return `${name}${int8}${dyn}.onnx`;
}

/** The encoder is only needed to clone a voice from audio. */
export function requiredModelStems(voiceCloning = true): ModelStem[] {
  return [...(voiceCloning ? (["mimi_encoder"] as ModelStem[]) : []), "text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder"];
}

/**
 * A bundle with `insert_bos_before_voice` (German and English at the pinned
 * revision) must have its BOS asset; returns whether BOS will be inserted.
 */
export function requireBos(
  metadata: { language?: string; insert_bos_before_voice?: boolean },
  bos: unknown,
): boolean {
  if (metadata.insert_bos_before_voice && !bos) {
    throw new Error(`${metadata.language ?? "This"} bundle requires bos_before_voice.npy, but the BOS asset is missing or could not be loaded.`);
  }
  return Boolean(metadata.insert_bos_before_voice && bos);
}

/** @deprecated Use requireBos(); it now checks every language. */
export const requireGermanBos = requireBos;

/** mulberry32: small, fast and identical on every JS engine. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
