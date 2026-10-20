// Measurement helpers used by the benchmark and examples.

export interface GenerationRecord {
  generationSeconds: number;
  audioSeconds: number;
  /** Real-time factor: wall seconds per audio second (< 1 is faster than real time). */
  rtf: number;
  firstChunkSeconds: number | null;
}

/** RTF from elapsed wall time and the samples actually returned (never the engine's own estimate). */
export function makeRecord({
  elapsedMs,
  firstChunkMs = null,
  sampleCount,
  sampleRate,
}: {
  elapsedMs: number;
  firstChunkMs?: number | null;
  sampleCount: number;
  sampleRate: number;
}): GenerationRecord {
  if (![elapsedMs, sampleCount, sampleRate].every(Number.isFinite) || elapsedMs <= 0 || sampleCount <= 0 || sampleRate <= 0) {
    throw new Error("Invalid generation measurement");
  }
  const audioSeconds = sampleCount / sampleRate;
  return {
    generationSeconds: elapsedMs / 1000,
    audioSeconds,
    rtf: elapsedMs / 1000 / audioSeconds,
    firstChunkSeconds: firstChunkMs === null ? null : firstChunkMs / 1000,
  };
}

export function median(values: readonly number[]): number | null {
  if (!values.length) return null;
  const a = [...values].sort((x, y) => x - y);
  const i = a.length >> 1;
  return a.length % 2 ? a[i] : (a[i - 1] + a[i]) / 2;
}

/** Medians over measured rows; warm-ups and failures are counted, not averaged. */
export function summarize(rows: readonly { warmup?: boolean; error?: unknown; rtf?: number; firstChunkSeconds?: number | null }[]) {
  const valid = rows.filter((r) => !r.warmup && !r.error && Number.isFinite(r.rtf));
  return {
    n: valid.length,
    medianRtf: median(valid.map((r) => r.rtf!)),
    medianFirstChunkSeconds: median(valid.map((r) => r.firstChunkSeconds).filter((x): x is number => Number.isFinite(x))),
    errors: rows.filter((r) => r.error).length,
    warmups: rows.filter((r) => r.warmup).length,
  };
}

/**
 * Request-to-playback latency, pairing the wall clock and the audio clock at
 * scheduling time (a freshly resumed audio clock may lag wall time). This is
 * the scheduled start, not a speaker measurement.
 */
export function startLatency({
  requestStartedMs,
  generationStartedMs,
  snapshotPerformanceMs,
  snapshotAudioSeconds,
  scheduledStartSeconds,
}: {
  requestStartedMs: number;
  generationStartedMs: number;
  snapshotPerformanceMs: number;
  snapshotAudioSeconds: number;
  scheduledStartSeconds: number;
}) {
  const all = [requestStartedMs, generationStartedMs, snapshotPerformanceMs, snapshotAudioSeconds, scheduledStartSeconds];
  if (!all.every(Number.isFinite) || generationStartedMs < requestStartedMs || snapshotPerformanceMs < generationStartedMs) {
    throw new Error("Invalid playback clock boundaries");
  }
  const scheduledStartMs = snapshotPerformanceMs + (scheduledStartSeconds - snapshotAudioSeconds) * 1000;
  if (scheduledStartMs < generationStartedMs) throw new Error("Invalid playback clock boundaries");
  const requestToPlaybackSeconds = (scheduledStartMs - requestStartedMs) / 1000;
  return {
    requestToPlaybackSeconds,
    playbackStartSeconds: (scheduledStartMs - generationStartedMs) / 1000,
    underOneSecond: requestToPlaybackSeconds <= 1,
  };
}

/** Simulated live playback of chunk arrival times: chunks start on arrival, then back-to-back. */
export function simulateLivePlayback(chunks: readonly { seconds: number; arrival: number }[], toleranceSeconds = 0.02) {
  let next: number | null = null;
  let gaps = 0;
  let gapSeconds = 0;
  for (const c of chunks) {
    if (next === null) next = c.arrival;
    const gap = c.arrival - next;
    if (gap > toleranceSeconds) {
      gaps++;
      gapSeconds += gap;
    }
    next = Math.max(next, c.arrival) + c.seconds;
  }
  return { gaps, gapSeconds };
}
