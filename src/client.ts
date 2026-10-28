// Main-thread API. All inference runs in a module Web Worker, so synthesis
// never blocks the UI. Importing this module has no side effects (safe during
// SSR); the worker starts on load().

import {
  estimateSpeechSeconds,
  prepareReferenceAudio,
  StreamPlayer,
  type AudioContextLike,
  type ChunkMeta,
  type GenerationMetrics,
  type StreamPlayerOptions,
  type VoiceEmbedding,
} from "./core/index.js";
import type { ProgressEvent } from "./host.js";
import { MANIFESTS, type Precision, type StockVoice } from "./manifest.js";
import { BUILD_ID, type ReadyInfo, type WorkerConfig, type WorkerEvent } from "./protocol.js";

/** onnxruntime-web version the runtime is tested with (pinned, never "latest"). */
export const ORT_VERSION = "1.20.0";
export const DEFAULT_ORT_BASE_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

export interface PocketTTSOptions {
  /** Language bundle: "german" (default) or "english". */
  language?: string;
  /** "int8-wide" (default, smallest download), "int8-core" or "fp32". */
  precision?: Precision;
  /** Folder with bundle.json, tokenizer and graphs. Default: the pinned Hugging Face revision. */
  modelBaseUrl?: string;
  /** Folder with stock voice files (`voices/...`). Default: modelBaseUrl. */
  voiceBaseUrl?: string;
  /** onnxruntime-web dist folder (ort.wasm.min.mjs + .wasm files). Default: jsDelivr, pinned. */
  ortBaseUrl?: string;
  /** WASM threads when cross-origin isolated. Default 4 (6 was slower than 4 in our runs). */
  maxThreads?: number;
  /** Persist downloads in Cache Storage, keyed by model revision. Default true. */
  cache?: boolean;
  /**
   * Size KV caches to the context instead of 1000 fixed slots (the graphs are
   * patched in memory at load). Same audio, far less copying. Default true.
   */
  dynamicCache?: boolean;
  temperature?: number;
  seed?: number;
  /** Override the worker script URL (e.g. a self-hosted copy of dist/worker.js). */
  workerUrl?: string | URL;
}

export type LoadEvent = ({ type: "progress" } & ProgressEvent) | { type: "status"; status: string; stem?: string; filename?: string };

export interface GenerateRequest {
  /** Voice reference from useVoice(), loadVoiceEmbedding() or cloneVoice(). */
  voice: string;
  onChunk?: (audio: Float32Array, meta: ChunkMeta) => void;
  /** Decoder block size after the first block. Default 12; speak() uses 3. */
  deliveryFrames?: number;
  firstChunkFrames?: number;
  maxFrames?: number;
}

export type PlaybackMode = "queued" | "adaptive" | "immediate" | "complete";

export interface SpeakRequest {
  voice: string;
  audioContext?: AudioContext | AudioContextLike;
  /**
   * "queued" (default): start after `primeSeconds` (default 1 s) of audio.
   * "adaptive": start as soon as the measured generation speed predicts no gaps.
   * "immediate": start on the first block. "complete": wait for the whole utterance.
   */
  mode?: PlaybackMode;
  primeSeconds?: number;
  deliveryFrames?: number;
  onChunk?: (audio: Float32Array, meta: ChunkMeta) => void;
  /** Called once when playback is scheduled. */
  onPlaybackStart?: (player: StreamPlayer) => void;
  player?: Omit<StreamPlayerOptions, "sampleRate" | "adaptive" | "primeSeconds" | "waitForComplete">;
}

export interface SpeakResult {
  metrics: GenerationMetrics;
  player: StreamPlayer;
  /** Raw generated audio, including inserted pauses between text chunks. */
  chunks: Float32Array[];
  /** Resolves when playback has finished. */
  played: Promise<void>;
}

export type { ReadyInfo };

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
}

export class PocketTTS {
  readonly config: WorkerConfig;
  private workerUrl: string | URL | undefined;
  private worker: Worker | null = null;
  info: ReadyInfo | null = null;
  private nextId = 1;
  private pending = new Map<number, PendingRequest>();
  private onChunk: GenerateRequest["onChunk"] | null = null;
  private loadListeners = new Set<(e: LoadEvent) => void>();
  private loadPromise: Promise<ReadyInfo> | null = null;
  private cloneCounter = 0;
  private generating = false;
  private recentRates: number[] = [];
  private activePlayer: StreamPlayer | null = null;

  constructor(options: PocketTTSOptions = {}) {
    const language = options.language ?? "german";
    if (!MANIFESTS[language]) throw new Error(`Unsupported language: ${language}. Supported: ${Object.keys(MANIFESTS).join(", ")}`);
    const precision = options.precision ?? "int8-wide";
    if (!(precision in MANIFESTS[language].precisions)) throw new Error(`Unknown precision: ${precision}`);
    this.config = {
      language,
      precision,
      modelBaseUrl: options.modelBaseUrl?.replace(/\/$/, "") ?? null,
      voiceBaseUrl: options.voiceBaseUrl?.replace(/\/$/, "") ?? null,
      ortBaseUrl: options.ortBaseUrl ?? DEFAULT_ORT_BASE_URL,
      dynamicCache: options.dynamicCache !== false,
      temperature: options.temperature ?? 0.3,
      seed: options.seed ?? 42,
      maxThreads: options.maxThreads ?? 4,
      cache: options.cache !== false,
    };
    this.workerUrl = options.workerUrl;
  }

  get ready(): boolean {
    return this.info !== null && this.worker !== null;
  }

  get sampleRate(): number {
    return this.info?.sampleRate ?? MANIFESTS[this.config.language].sampleRate;
  }

  /** Stock voices available for this language. */
  get voices(): readonly StockVoice[] {
    return MANIFESTS[this.config.language].voices;
  }

  get isGenerating(): boolean {
    return this.generating;
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const worker = this.workerUrl
      ? new Worker(this.workerUrl, { type: "module" })
      : new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.worker = worker;
    worker.onmessage = (e: MessageEvent<WorkerEvent>) => {
      if (this.worker === worker) this.handle(e.data);
    };
    worker.onerror = (e: ErrorEvent) => {
      if (this.worker === worker) this.reset(new Error(e.message || "Worker error"));
    };
    return worker;
  }

  // Reject everything outstanding and drop the worker; load() can start fresh.
  private reset(err: Error) {
    for (const { reject } of this.pending.values()) reject(err);
    this.pending.clear();
    this.activePlayer?.stop();
    this.worker?.terminate();
    this.worker = null;
    this.info = null;
    this.loadPromise = null;
    this.onChunk = null;
    this.generating = false;
  }

  private handle(msg: WorkerEvent) {
    switch (msg.type) {
      case "chunk":
        this.onChunk?.(msg.audio, msg.meta);
        return;
      case "progress":
      case "status":
        for (const listener of this.loadListeners) listener(msg as LoadEvent);
        return;
      case "result":
      case "error": {
        const p = this.pending.get(msg.id);
        if (!p) return;
        this.pending.delete(msg.id);
        if (msg.type === "result") p.resolve(msg.result);
        else p.reject(new Error(msg.error));
        return;
      }
    }
  }

  private request<T>(type: string, payload: unknown, transfer: Transferable[] = []): Promise<T> {
    const worker = this.ensureWorker();
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      worker.postMessage({ id, type, payload }, transfer);
    });
  }

  private assertReady(operation: string) {
    if (!this.ready) throw new Error(`Call load() successfully before ${operation}.`);
  }

  /**
   * Download the runtime and graphs (or read them from Cache Storage) and
   * create the sessions. Concurrent calls share one load; after success it
   * resolves immediately. Voices can then be switched without reloading.
   */
  load(onEvent?: (e: LoadEvent) => void): Promise<ReadyInfo> {
    if (onEvent) this.loadListeners.add(onEvent);
    if (!this.loadPromise) {
      const attempt: Promise<ReadyInfo> = this.request<ReadyInfo>("init", this.config)
        .then((info) => {
          if (info.build !== BUILD_ID) {
            throw new Error(`Worker build ${info.build} does not match client build ${BUILD_ID}; a stale worker.js is cached or mixed versions are deployed.`);
          }
          this.info = info;
          return info;
        })
        .catch((err: Error) => {
          // A failed init leaves a half-initialized worker: start clean next time.
          if (this.loadPromise === attempt) this.reset(err);
          throw err;
        });
      this.loadPromise = attempt;
    }
    const p = this.loadPromise;
    if (onEvent) p.finally(() => this.loadListeners.delete(onEvent)).catch(() => {});
    return p;
  }

  /** Prepare a stock voice (downloads its small embedding once; no encoder needed). */
  async useVoice(id: string): Promise<string> {
    this.assertReady("useVoice()");
    return (await this.request<{ ref: string }>("useVoice", { id })).ref;
  }

  /**
   * Encode a reference clip locally (the audio never leaves the browser). The
   * encoder (~40 MB) loads on first use. Keep the result and pass it to
   * loadVoiceEmbedding() later: it works with every precision.
   */
  async encodeVoice(audio: Float32Array, opts: { inputSampleRate?: number } = {}): Promise<VoiceEmbedding> {
    this.assertReady("encodeVoice()");
    const pcm = prepareReferenceAudio(audio, { inputSampleRate: opts.inputSampleRate, targetRate: this.sampleRate });
    return this.request<VoiceEmbedding>("encodeVoice", { audio: pcm }, [pcm.buffer]);
  }

  /** Prepare a voice from an encodeVoice() result. */
  async loadVoiceEmbedding(embedding: VoiceEmbedding, name?: string): Promise<string> {
    this.assertReady("loadVoiceEmbedding()");
    const data = embedding.data.slice();
    const ref = name ?? `clone:${++this.cloneCounter}`;
    return (await this.request<{ ref: string }>("loadVoiceEmbedding", { ref, data, shape: [...embedding.shape] }, [data.buffer])).ref;
  }

  /** encodeVoice() + loadVoiceEmbedding(). */
  async cloneVoice(audio: Float32Array, opts: { inputSampleRate?: number; name?: string } = {}): Promise<string> {
    return this.loadVoiceEmbedding(await this.encodeVoice(audio, opts), opts.name);
  }

  /** Synthesize speech, streaming mono Float32 chunks at `sampleRate` to onChunk. */
  async generate(text: string, opts: GenerateRequest): Promise<GenerationMetrics> {
    this.assertReady("generate()");
    if (!opts?.voice) throw new Error("generate() requires a `voice` reference.");
    const deliveryFrames = opts.deliveryFrames ?? 12;
    if (!Number.isInteger(deliveryFrames) || deliveryFrames < 1 || deliveryFrames > 64) throw new Error("Unsupported delivery frame count.");
    if (this.generating) throw new Error("A generation is already in progress.");
    this.generating = true;
    this.onChunk = opts.onChunk ?? null;
    const started = now();
    try {
      const { metrics } = await this.request<{ metrics: GenerationMetrics }>("generate", {
        text,
        voiceRef: opts.voice,
        deliveryFrames,
        firstChunkFrames: opts.firstChunkFrames,
        maxFrames: opts.maxFrames,
      });
      const wall = (now() - started) / 1000;
      if (!metrics.stopped && metrics.audioDuration > 0 && wall > 0) {
        this.recentRates = [...this.recentRates, metrics.audioDuration / wall].slice(-3);
      }
      return metrics;
    } finally {
      this.generating = false;
      this.onChunk = null;
    }
  }

  /** Slowest recent generation speed (audio s per wall s), used by the adaptive start. */
  get rateHint(): number | undefined {
    return this.recentRates.length ? Math.min(...this.recentRates) : undefined;
  }

  /** Generate and play through Web Audio. */
  async speak(text: string, req: SpeakRequest): Promise<SpeakResult> {
    this.assertReady("speak()");
    const context = req.audioContext ?? new AudioContext();
    if ("resume" in context && typeof context.resume === "function") await context.resume();
    const mode = req.mode ?? "queued";
    const player = new StreamPlayer(context, {
      sampleRate: this.sampleRate,
      leadSeconds: 0.05,
      ...req.player,
      primeSeconds: mode === "queued" ? (req.primeSeconds ?? 1) : (req.primeSeconds ?? 0),
      waitForComplete: mode === "complete",
      adaptive: mode === "adaptive" ? { estimatedSeconds: estimateSpeechSeconds(text), rateHint: this.rateHint } : null,
    });
    this.activePlayer?.stop();
    this.activePlayer = player;
    const chunks: Float32Array[] = [];
    let announced = false;
    const announce = () => {
      if (!announced && player.startedAt !== null) {
        announced = true;
        req.onPlaybackStart?.(player);
      }
    };
    try {
      const metrics = await this.generate(text, {
        voice: req.voice,
        deliveryFrames: req.deliveryFrames ?? 3,
        onChunk: (audio, meta) => {
          chunks.push(audio);
          req.onChunk?.(audio, meta);
          // The player receives a copy so `chunks` stays the untouched raw output.
          player.play(audio.slice(), meta);
          announce();
        },
      });
      const played = player.finish();
      announce();
      played.finally(() => {
        if (this.activePlayer === player) this.activePlayer = null;
      });
      return { metrics, player, chunks, played };
    } catch (err) {
      player.stop();
      if (this.activePlayer === player) this.activePlayer = null;
      throw err;
    }
  }

  /** Stop the current generation and speak() playback, including while still buffering. */
  async stop(): Promise<void> {
    this.activePlayer?.stop();
    this.activePlayer = null;
    if (!this.worker || !this.generating) return;
    await this.request("stop", {});
  }

  /** Terminate the worker and free all resources; outstanding calls reject. */
  destroy(): void {
    this.reset(new Error("PocketTTS instance destroyed."));
    this.loadListeners.clear();
  }

  /** Delete every cached model revision (forces a fresh download). */
  static async clearCache(): Promise<number> {
    if (typeof caches === "undefined") return 0;
    let n = 0;
    for (const name of await caches.keys()) if (name.startsWith("pocket-tts:") && (await caches.delete(name))) n++;
    return n;
  }

  /** Storage used by this origin and its quota, if the browser reports it. */
  static async storageEstimate(): Promise<{ usage: number; quota: number } | null> {
    if (typeof navigator === "undefined" || !navigator.storage?.estimate) return null;
    const { usage = 0, quota = 0 } = await navigator.storage.estimate();
    return { usage, quota };
  }
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

/** Decode any audio file the browser understands to mono Float32 at its native rate. Runs locally. */
export async function decodeAudioFile(input: Blob | ArrayBuffer, context?: BaseAudioContext): Promise<{ audio: Float32Array; sampleRate: number }> {
  const bytes = input instanceof Blob ? await input.arrayBuffer() : input.slice(0);
  const ctx = context ?? new OfflineAudioContext(1, 1, 48000);
  const buffer = await ctx.decodeAudioData(bytes);
  const audio = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const ch = buffer.getChannelData(c);
    for (let i = 0; i < ch.length; i++) audio[i] += ch[i] / buffer.numberOfChannels;
  }
  return { audio, sampleRate: buffer.sampleRate };
}
