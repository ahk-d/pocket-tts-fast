// Dynamic KV-cache sizing.
//
// The exported graphs declare every attention cache with 1000 slots, but they
// only read Slice[0 : step + len]. With a symbolic cache axis (patched graph)
// the caller can pass a buffer that holds just the filled prefix plus headroom,
// so each run copies kilobytes-to-a-few-MB instead of 49 MB.

/** Slots to allocate so `needed` fit, with headroom so growth is rare. */
export function cacheCapacityFor(needed: number, headroom = 32, maxSlots = 1000): number {
  return Math.min(maxSlots, Math.ceil((needed + headroom) / 64) * 64);
}

/**
 * Copy the first `used` slots along `axis` of a row-major cache into a buffer
 * with `capacity` slots; the rest is `fill`. The graphs only read filled slots.
 */
export function resizeCacheData(
  data: Float32Array,
  shape: readonly number[],
  axis: number,
  used: number,
  capacity: number,
  fill = NaN,
): Float32Array {
  const outer = shape.slice(0, axis).reduce((a, b) => a * b, 1);
  const inner = shape.slice(axis + 1).reduce((a, b) => a * b, 1);
  const keep = Math.min(used, shape[axis], capacity) * inner;
  const out = new Float32Array(outer * capacity * inner);
  if (fill !== 0) out.fill(fill);
  for (let o = 0; o < outer; o++) {
    const from = o * shape[axis] * inner;
    out.set(data.subarray(from, from + keep), o * capacity * inner);
  }
  return out;
}

export function withAxis(shape: readonly number[], axis: number, size: number): number[] {
  const out = shape.slice();
  out[axis] = size;
  return out;
}

/** Bytes moved per run for a cache list, for documentation and the benchmark. */
export function cacheBytes(shapes: readonly (readonly number[])[], bytesPerElement = 4): number {
  return shapes.reduce((sum, s) => sum + s.reduce((a, b) => a * b, 1) * bytesPerElement, 0);
}
