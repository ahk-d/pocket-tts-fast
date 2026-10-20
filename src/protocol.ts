import type { ChunkMeta, GenerationMetrics, VoiceEmbedding } from "./core/index.js";
import type { ProgressEvent } from "./host.js";
import type { Precision } from "./manifest.js";

/** Settings sent to the worker on init. */
export interface WorkerConfig {
  language: string;
  precision: Precision;
  modelBaseUrl: string | null;
  voiceBaseUrl: string | null;
  ortBaseUrl: string;
  dynamicCache: boolean;
  temperature: number;
  seed: number;
  maxThreads: number;
  cache: boolean;
}

declare const __POCKET_TTS_BUILD__: string;
/** Version + source hash of this build (set by scripts/build.mjs). */
export const BUILD_ID: string = typeof __POCKET_TTS_BUILD__ === "string" ? __POCKET_TTS_BUILD__ : "dev";

export interface ReadyInfo {
  /** Build of the worker that answered; must equal the client's BUILD_ID. */
  build: string;
  language: string;
  /** Manifest id: language and pinned model revision, e.g. "german@58a6d00". */
  modelId: string;
  precision: Precision;
  sampleRate: number;
  loadedFiles: string[];
  dynamicCache: boolean;
  requestedThreads: number;
  /** WASM threads actually configured (1 without cross-origin isolation). */
  actualThreads: number;
  crossOriginIsolated: boolean;
  executionProvider: "wasm";
  ortVersion: string | null;
}

export type WorkerRequest =
  | { id: number; type: "init"; payload: WorkerConfig }
  | { id: number; type: "useVoice"; payload: { id: string } }
  | { id: number; type: "encodeVoice"; payload: { audio: Float32Array } }
  | { id: number; type: "loadVoiceEmbedding"; payload: VoiceEmbedding & { ref: string } }
  | { id: number; type: "generate"; payload: { text: string; voiceRef: string; deliveryFrames: number; firstChunkFrames?: number; maxFrames?: number } }
  | { id: number; type: "stop"; payload?: undefined };

export type WorkerEvent =
  | { type: "status"; status: string; stem?: string; filename?: string }
  | ({ type: "progress" } & ProgressEvent)
  | { type: "chunk"; audio: Float32Array; meta: ChunkMeta }
  | { id: number; type: "result"; result: unknown }
  | { id: number; type: "error"; error: string };

export type { GenerationMetrics };
