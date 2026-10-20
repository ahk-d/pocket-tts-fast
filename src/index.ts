// @flowent/pocket-tts — fast, local Pocket TTS in the browser.
// Importing has no side effects; nothing downloads until load().

import { wavFromChunks } from "./core/index.js";

export { PocketTTS, decodeAudioFile, DEFAULT_ORT_BASE_URL, ORT_VERSION } from "./client.js";
export { BUILD_ID } from "./protocol.js";
export type { PocketTTSOptions, GenerateRequest, SpeakRequest, SpeakResult, PlaybackMode, LoadEvent, ReadyInfo } from "./client.js";
export { GERMAN, MANIFESTS, filesFor, downloadBytes } from "./manifest.js";
export type { Precision, Asset, StockVoice, LanguageManifest } from "./manifest.js";
export { createWebHost, cacheNameFor, pruneOldCaches, sha256Hex, AssetIntegrityError } from "./host.js";
export type { OrtWebModule, ProgressEvent, WebHostOptions } from "./host.js";
export {
  StreamPlayer,
  scheduledReplay,
  adaptiveStartSeconds,
  estimateSpeechSeconds,
  wavFromChunks,
  decodeWav,
  resampleLinear,
  prepareReferenceAudio,
  encodeNpyFloat32,
  parseNpyFloat32,
  PocketTTSEngine,
  makeCacheDynamic,
  dynamicCacheRules,
  cacheCapacityFor,
  resizeCacheData,
  splitIntoChunks,
  splitTextIntoSentences,
  normalizePunctuation,
  framesAfterEosForWordCount,
  prepareTextPrompt,
  SentencePieceTokenizer,
  PRECISION_STEMS,
  modelFilename,
  requiredModelStems,
  requireGermanBos,
  createSeededRandom,
  validateEmbedding,
  valueInfoDims,
  makeRecord,
  summarize,
  median,
  startLatency,
  simulateLivePlayback,
  utf8Encode,
  utf8Decode,
} from "./core/index.js";
export type { ChunkMeta, GenerationMetrics, TextChunkReport, ChunkEndReason, VoiceEmbedding, StreamPlayerOptions, AudioContextLike, EngineHost } from "./core/index.js";

/** WAV Blob (16-bit PCM) from generated chunks. */
export function chunksToWavBlob(chunks: readonly Float32Array[], sampleRate = 24000): Blob {
  return new Blob([wavFromChunks(chunks, sampleRate).buffer], { type: "audio/wav" });
}
