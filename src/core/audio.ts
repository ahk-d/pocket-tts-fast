/** Linear resampler for reference clips (same rule as pocket-tts-js). */
export function resampleLinear(data: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (sourceRate === targetRate) return data;
  const ratio = sourceRate / targetRate;
  const outLength = Math.floor(data.length / ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcIndex = i * ratio;
    const floor = Math.floor(srcIndex);
    const ceil = Math.min(floor + 1, data.length - 1);
    const t = srcIndex - floor;
    out[i] = data[floor] * (1 - t) + data[ceil] * t;
  }
  return out;
}

/** Mono PCM at `targetRate`, capped to `maxSeconds` (the encoder sees at most 10 s). */
export function prepareReferenceAudio(
  audio: Float32Array,
  { inputSampleRate, targetRate = 24000, maxSeconds = 10 }: { inputSampleRate?: number; targetRate?: number; maxSeconds?: number } = {},
): Float32Array {
  let pcm = inputSampleRate && inputSampleRate !== targetRate ? resampleLinear(audio, inputSampleRate, targetRate) : audio;
  const maxSamples = Math.floor(targetRate * maxSeconds);
  if (pcm.length > maxSamples) pcm = pcm.slice(0, maxSamples);
  return pcm === audio ? audio.slice() : pcm;
}

/**
 * 16-bit PCM WAV bytes from mono Float32 chunks. Throws on empty, silent or
 * non-finite audio, so a broken generation can't pass as a result.
 */
export function wavFromChunks(chunks: readonly Float32Array[], rate: number): { buffer: ArrayBuffer; sampleCount: number; peak: number } {
  const sampleCount = chunks.reduce((n, a) => n + a.length, 0);
  if (!sampleCount || !Number.isFinite(rate) || rate <= 0) throw new Error("No valid audio samples");
  const buffer = new ArrayBuffer(44 + sampleCount * 2);
  const v = new DataView(buffer);
  const text = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(offset + i, s.charCodeAt(i));
  };
  text(0, "RIFF");
  v.setUint32(4, 36 + sampleCount * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  text(36, "data");
  v.setUint32(40, sampleCount * 2, true);
  let offset = 44;
  let peak = 0;
  for (const chunk of chunks) {
    for (const sample of chunk) {
      if (!Number.isFinite(sample)) throw new Error("Non-finite generated audio");
      peak = Math.max(peak, Math.abs(sample));
      v.setInt16(offset, Math.round(Math.max(-1, Math.min(1, sample)) * 32767), true);
      offset += 2;
    }
  }
  if (!peak) throw new Error("Silent generated audio");
  return { buffer, sampleCount, peak };
}

/**
 * Decode a 16/24/32-bit PCM or 32-bit float WAV to mono Float32 (channels averaged).
 * For other formats use the platform's decoder (e.g. AudioContext.decodeAudioData).
 */
export function decodeWav(input: ArrayBuffer | Uint8Array): { audio: Float32Array; sampleRate: number } {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (o: number) => String.fromCharCode(bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]);
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("Not a RIFF/WAVE file");
  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const id = tag(offset);
    const size = v.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      format = v.getUint16(body, true);
      channels = v.getUint16(body + 2, true);
      sampleRate = v.getUint32(body + 4, true);
      bits = v.getUint16(body + 14, true);
      if (format === 0xfffe && size >= 26) format = v.getUint16(body + 24, true); // WAVE_FORMAT_EXTENSIBLE
    } else if (id === "data") {
      if (!channels) throw new Error("WAV data before fmt chunk");
      const bytesPer = bits / 8;
      const end = Math.min(body + size, bytes.length);
      const frames = Math.floor((end - body) / (bytesPer * channels));
      const audio = new Float32Array(frames);
      for (let f = 0; f < frames; f++) {
        let sum = 0;
        for (let c = 0; c < channels; c++) {
          const p = body + (f * channels + c) * bytesPer;
          if (format === 3 && bits === 32) sum += v.getFloat32(p, true);
          else if (format === 1 && bits === 16) sum += v.getInt16(p, true) / 32768;
          else if (format === 1 && bits === 24) sum += ((bytes[p] | (bytes[p + 1] << 8) | (bytes[p + 2] << 16)) << 8 >> 8) / 8388608;
          else if (format === 1 && bits === 32) sum += v.getInt32(p, true) / 2147483648;
          else throw new Error(`Unsupported WAV format ${format}/${bits}-bit`);
        }
        audio[f] = sum / channels;
      }
      return { audio, sampleRate };
    }
    offset = body + size + (size & 1);
  }
  throw new Error("WAV file has no data chunk");
}

/**
 * Audio to hold before starting playback so it never overtakes generation.
 *
 * Playback drains 1 s/s while generation adds `rate` s/s. With R seconds still
 * to generate the queue bottoms out after R/rate seconds, so playback must
 * start with at least (1/rate - 1)·R buffered. The rate is discounted by
 * `margin` for speed dips, and the floor (one delivery block) absorbs
 * scheduling jitter. When generation is faster than real time (after the
 * margin) only the floor is needed.
 */
export function adaptiveStartSeconds({
  rate,
  receivedSeconds,
  estimatedSeconds,
  margin = 1.2,
  floorSeconds = 0.24,
}: {
  rate: number;
  receivedSeconds: number;
  estimatedSeconds: number;
  margin?: number;
  floorSeconds?: number;
}): number {
  if (!(rate > 0)) return Infinity;
  const effective = rate / margin;
  const remaining = Math.max(0, estimatedSeconds - receivedSeconds);
  return floorSeconds + (effective >= 1 ? 0 : (1 / effective - 1) * remaining);
}

/** Generous speech-length guess (tuned on German); overestimating only delays the start. */
export function estimateSpeechSeconds(text: string, secondsPerChar = 0.06): number {
  return 0.6 + secondsPerChar * text.trim().length;
}
