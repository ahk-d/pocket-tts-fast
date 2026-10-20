// Compiled (not run) against dist/*.d.ts by `npm run test:types`: the public
// API a TypeScript consumer sees.
import {
  PocketTTS,
  chunksToWavBlob,
  GERMAN,
  type ChunkMeta,
  type GenerationMetrics,
  type Precision,
  type ReadyInfo,
  type SpeakResult,
  type VoiceEmbedding,
} from "../../dist/index.js";

async function usage(): Promise<void> {
  const precision: Precision = "int8-wide";
  const tts = new PocketTTS({ precision, maxThreads: 4, voiceBaseUrl: "/models/german" });
  const info: ReadyInfo = await tts.load((e) => {
    if (e.type === "progress") console.log(e.label, e.loaded / e.total, e.fromCache);
  });
  const threads: number = info.actualThreads;
  const voice: string = await tts.useVoice(GERMAN.voices[0].id);
  const chunks: Float32Array[] = [];
  const metrics: GenerationMetrics = await tts.generate("Hallo.", {
    voice,
    deliveryFrames: 3,
    onChunk: (audio: Float32Array, meta: ChunkMeta) => {
      if (!meta.isSilence) chunks.push(audio);
    },
  });
  const rtf: number = metrics.genTime / metrics.audioDuration;
  const result: SpeakResult = await tts.speak("Tschüss.", { voice, mode: "queued", primeSeconds: 1 });
  await result.played;
  const embedding: VoiceEmbedding = await tts.encodeVoice(new Float32Array(48000), { inputSampleRate: 48000 });
  const cloned: string = await tts.loadVoiceEmbedding(embedding);
  const blob: Blob = chunksToWavBlob(chunks, tts.sampleRate);
  await tts.stop();
  tts.destroy();
  void [threads, rtf, cloned, blob];

  // @ts-expect-error unknown precision names are rejected at compile time
  new PocketTTS({ precision: "int4" });
  // @ts-expect-error generate() requires a voice
  await tts.generate("Hallo.", {});
}
void usage;
