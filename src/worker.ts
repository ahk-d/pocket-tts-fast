// Inference worker. Runs the engine off the main thread with onnxruntime-web
// (WASM), which it imports at runtime from `ortBaseUrl` (a pinned version), so
// the package stays small and needs no bundler configuration.

import { PocketTTSEngine, PRECISION_STEMS } from "./core/index.js";
import { createWebHost, pruneOldCaches, type OrtWebModule } from "./host.js";
import { MANIFESTS } from "./manifest.js";
import { BUILD_ID, type ReadyInfo, type WorkerConfig, type WorkerEvent, type WorkerRequest } from "./protocol.js";

interface WorkerScope {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: { data: WorkerRequest }) => void) | null;
  crossOriginIsolated?: boolean;
}

const scope = self as unknown as WorkerScope;

let ort: OrtWebModule | null = null;
let engine: PocketTTSEngine | null = null;
let config: WorkerConfig | null = null;
let threads = 1;

function post(msg: WorkerEvent, transfer: Transferable[] = []) {
  scope.postMessage(msg, transfer);
}

async function loadOrt(cfg: WorkerConfig): Promise<OrtWebModule> {
  if (ort) return ort;
  post({ type: "status", status: "loading-runtime" });
  const base = cfg.ortBaseUrl.endsWith("/") ? cfg.ortBaseUrl : `${cfg.ortBaseUrl}/`;
  const mod = await import(/* webpackIgnore: true */ /* @vite-ignore */ `${base}ort.wasm.min.mjs`);
  const loaded = (mod.default ?? mod) as OrtWebModule;
  loaded.env.wasm.wasmPaths = base;
  loaded.env.wasm.simd = true;
  // Threads need cross-origin isolation (COOP/COEP); otherwise WASM runs on one thread.
  const hw = (globalThis as { navigator?: { hardwareConcurrency?: number } }).navigator?.hardwareConcurrency || 4;
  threads = scope.crossOriginIsolated ? Math.max(1, Math.min(hw, cfg.maxThreads)) : 1;
  loaded.env.wasm.numThreads = threads;
  ort = loaded;
  return loaded;
}

async function init(cfg: WorkerConfig): Promise<ReadyInfo> {
  const manifest = MANIFESTS[cfg.language];
  if (!manifest) throw new Error(`Unsupported language: ${cfg.language}. Supported: ${Object.keys(MANIFESTS).join(", ")}`);
  if (!(cfg.precision in PRECISION_STEMS)) throw new Error(`Unknown precision: ${cfg.precision}`);
  const ortModule = await loadOrt(cfg);
  if (cfg.cache) pruneOldCaches(manifest).catch(() => {});
  const host = createWebHost({
    ort: ortModule,
    manifest,
    modelBaseUrl: cfg.modelBaseUrl ?? undefined,
    voiceBaseUrl: cfg.voiceBaseUrl ?? undefined,
    cache: cfg.cache,
    onProgress: (p) => post({ type: "progress", ...p }),
  });
  engine = await PocketTTSEngine.load(host, {
    quantizedStems: PRECISION_STEMS[cfg.precision],
    dynamicCache: cfg.dynamicCache ? "patch" : false,
    temperature: cfg.temperature,
    seed: cfg.seed,
    onStatus: (s) => post({ type: "status", ...s }),
  });
  config = cfg;
  return {
    build: BUILD_ID,
    language: manifest.language,
    modelId: manifest.id,
    precision: cfg.precision,
    sampleRate: engine.sampleRate,
    loadedFiles: engine.info.loadedModelFiles,
    dynamicCache: cfg.dynamicCache,
    requestedThreads: cfg.maxThreads,
    actualThreads: threads,
    crossOriginIsolated: Boolean(scope.crossOriginIsolated),
    executionProvider: "wasm",
    ortVersion: ortModule.env.versions?.web ?? null,
  };
}

function requireEngine(): PocketTTSEngine {
  if (!engine) throw new Error("Engine is not loaded.");
  return engine;
}

async function useVoice(id: string): Promise<string> {
  const e = requireEngine();
  const voice = MANIFESTS[config!.language].voices.find((v) => v.id === id);
  if (!voice) throw new Error(`Unknown stock voice: ${id}`);
  return e.loadEmbeddingAsset(voice.file, `stock:${id}`);
}

scope.onmessage = async (e) => {
  const req = e.data;
  const { id } = req;
  try {
    let result: unknown;
    const transfer: Transferable[] = [];
    switch (req.type) {
      case "init":
        result = await init(req.payload);
        break;
      case "useVoice":
        result = { ref: await useVoice(req.payload.id) };
        break;
      case "encodeVoice": {
        const embedding = await requireEngine().encodeVoice(req.payload.audio);
        transfer.push(embedding.data.buffer);
        result = embedding;
        break;
      }
      case "loadVoiceEmbedding": {
        const { data, shape, ref } = req.payload;
        result = { ref: await requireEngine().loadVoiceEmbedding({ data, shape }, ref) };
        break;
      }
      case "generate": {
        const { text, voiceRef, deliveryFrames, firstChunkFrames, maxFrames } = req.payload;
        result = {
          metrics: await requireEngine().generate(text, voiceRef, {
            deliveryFrames,
            firstChunkFrames,
            maxFrames: maxFrames ?? undefined,
            onChunk: (audio, meta) => post({ type: "chunk", audio, meta }, [audio.buffer]),
          }),
        };
        break;
      }
      case "stop":
        engine?.stop();
        result = { ok: true };
        break;
    }
    post({ id, type: "result", result }, transfer);
  } catch (err) {
    post({ id, type: "error", error: err instanceof Error ? err.message : String(err) });
  }
};
