// The Pocket TTS generation loop, written against the small EngineHost
// interface so the same code runs in a browser worker (onnxruntime-web), in
// Node, and in React Native (onnxruntime-react-native).
//
// Ported from vlapky/pocket-tts-js (MIT) with dynamic KV-cache sizing: with
// `dynamicCache` the FlowLM caches hold only the filled prefix plus headroom
// instead of 1000 slots, and the Mimi decoder's passthrough caches are empty.
// The graphs never read past the filled prefix, so the audio is bit-identical.

import { parseNpyFloat32, type Float32Tensor } from "./binary.js";
import { cacheCapacityFor, resizeCacheData, withAxis } from "./cache.js";
import { dynamicCacheRules, makeCacheDynamic } from "./onnx-patch.js";
import {
  createSeededRandom,
  DYNAMIC_CACHE_STEMS,
  modelFilename,
  PRECISION_STEMS,
  requireGermanBos,
  requiredModelStems,
  type ModelStem,
} from "./options.js";
import { createYielder, type EngineHost, type InferenceSessionLike, type TensorData, type TensorLike, type TensorType, type Yielder } from "./session.js";
import { countWords, framesAfterEosForWordCount, splitIntoChunks } from "./text.js";
import { SentencePieceTokenizer } from "./tokenizer.js";
import { utf8Decode } from "./utf8.js";

export interface StateEntry {
  input_name: string;
  output_name: string;
  module: string;
  key: string;
  shape: number[];
  dtype: TensorType;
  fill: "nan" | "zeros" | "ones" | string;
}

export interface BundleMetadata {
  language?: string;
  sample_rate: number;
  samples_per_frame: number;
  latent_dim: number;
  conditioning_dim: number;
  max_token_per_chunk?: number;
  tokenizer_file: string;
  bos_before_voice_file?: string;
  insert_bos_before_voice?: boolean;
  remove_semicolons?: boolean;
  model_recommended_frames_after_eos?: number | null;
  pad_with_spaces_for_short_inputs?: boolean;
  predefined_voices?: string[];
  flow_lm_state_manifest: StateEntry[];
  mimi_state_manifest: StateEntry[];
}

/** How KV caches are sized. */
export type DynamicCacheMode =
  /** Load the stock graphs and patch the cache axis in memory (default). */
  | "patch"
  /** Load `*_dyncache.onnx` graphs written by tools/make_dynamic_cache.py. */
  | "prebuilt"
  /** Stock behaviour: fixed 1000-slot caches copied on every call. */
  | false;

export interface EngineOptions {
  /** Graphs to load as INT8. Default: PRECISION_STEMS["int8-wide"]. */
  quantizedStems?: readonly string[];
  dynamicCache?: DynamicCacheMode;
  /** Load the voice encoder during load(); otherwise it loads on first encodeVoice(). */
  voiceCloning?: boolean;
  /** Gaussian latent sampling temperature. Default 0.3. */
  temperature?: number;
  /** Noise seed, reset for every generate() call. Default 42. */
  seed?: number;
  /** Extra cache slots allocated when a cache grows. Default 32. */
  cacheHeadroom?: number;
  /** Runs between frames so a stop request can be handled. */
  yieldToEventLoop?: Yielder;
  now?: () => number;
  onStatus?: (status: EngineStatus) => void;
}

export type EngineStatus =
  | { status: "loading-bundle" }
  | { status: "loading-model"; stem: ModelStem; filename: string }
  | { status: "ready" };

export interface VoiceEmbedding {
  data: Float32Array;
  shape: number[];
}

export interface ChunkMeta {
  chunkDuration: number;
  genTimeSec?: number;
  isFirst: boolean;
  isLast: boolean;
  /** First audio block of a text chunk. */
  chunkStart?: boolean;
  /** Pause inserted between text chunks. */
  isSilence: boolean;
}

export interface GenerateOptions {
  /** Decoder block size after the first block (smaller = earlier audio). Default 12. */
  deliveryFrames?: number;
  /** Size of the very first decoded block. Default 3. */
  firstChunkFrames?: number;
  /** Stop after this many frames (warm-up runs). */
  maxFrames?: number;
  onChunk?: (audio: Float32Array, meta: ChunkMeta) => void;
}

/** Why a text chunk stopped generating. Only "eos" is a complete chunk. */
export type ChunkEndReason = "eos" | "cancelled" | "frame-budget" | "frame-cap";

export interface TextChunkReport {
  index: number;
  text: string;
  tokenCount: number;
  /** Frames generated after the model's end-of-speech signal. */
  framesAfterEos: number;
  /** Seconds from generate() to: chunk start, conditioning done, first audio, chunk done. */
  startSeconds: number;
  conditioningSeconds: number;
  firstAudioSeconds: number | null;
  completionSeconds: number;
  generatedFrames: number;
  /** Frames actually decoded and delivered as audio. */
  emittedFrames: number;
  endReason: ChunkEndReason;
}

export interface GenerationMetrics {
  rtfx: number;
  genTime: number;
  audioDuration: number;
  stopped: boolean;
  flowLmSeconds: number;
  decoderSeconds: number;
  frames: number;
  deliveryFrames: number;
  /** True only if every text chunk ended by end-of-speech (nothing cancelled or capped). */
  complete: boolean;
  /** Per text chunk diagnostics, for finding missing or truncated output. */
  textChunks: TextChunkReport[];
}

export interface BundleInfo {
  language?: string;
  sampleRate: number;
  samplesPerFrame: number;
  predefinedVoices: string[];
  loadedModelFiles: string[];
  dynamicCache: DynamicCacheMode;
}

const CHUNK_GAP_SEC = 0.25;
const MAX_FRAMES = 500;
const LSD_STEPS = 1;
const EOS_THRESHOLD = -4.0;
// FlowLM caches are [2, 1, slots, 16, 64]; Mimi's passthrough caches are [2, 1, 8, slots, 64].
const FLOW_CACHE_AXIS = 2;
const MIMI_CACHE_AXIS = 3;

type State = Record<string, TensorLike>;

function makeFilledArray(shape: readonly number[], dtype: TensorType, fill: string): TensorData {
  const size = shape.reduce((a, b) => a * b, 1);
  if (dtype === "int64") return new BigInt64Array(size);
  if (dtype === "bool") return new Uint8Array(size);
  const data = new Float32Array(size);
  if (fill === "nan") data.fill(NaN);
  else if (fill === "ones") data.fill(1);
  return data;
}

/** Reject embeddings that cannot have come from this bundle's encoder. */
export function validateEmbedding(embedding: VoiceEmbedding, dim: number): void {
  const { data, shape } = embedding;
  if (!(data instanceof Float32Array)) throw new Error("Voice embedding data must be a Float32Array.");
  if (shape.length !== 3 || shape[0] !== 1 || shape[2] !== dim || !(shape[1] > 0)) {
    throw new Error(`Voice embedding shape [${shape}] does not match this model (expected [1, frames, ${dim}]).`);
  }
  if (data.length !== shape[1] * dim) throw new Error("Voice embedding data length does not match its shape.");
  for (let i = 0; i < data.length; i++) if (!Number.isFinite(data[i])) throw new Error("Voice embedding contains non-finite values.");
}

export class PocketTTSEngine {
  readonly host: EngineHost;
  readonly options: Required<Pick<EngineOptions, "temperature" | "seed" | "cacheHeadroom">> & EngineOptions;
  readonly dynamicCache: DynamicCacheMode;
  readonly quantizedStems: readonly string[];

  bundle!: BundleMetadata;
  tokenizer!: SentencePieceTokenizer;
  private bos: Float32Tensor | null = null;
  private sessions: Partial<Record<ModelStem, InferenceSessionLike>> = {};
  private encoderLoading: Promise<InferenceSessionLike> | null = null;
  private voiceStates = new Map<string, State>();
  private flowCacheNames: string[] = [];
  private flowStepName = "";
  private generating = false;
  private stopRequested = false;
  private yieldToEventLoop: Yielder;
  private now: () => number;
  private loadedModelFiles: string[] = [];

  private constructor(host: EngineHost, options: EngineOptions) {
    this.host = host;
    this.options = { temperature: 0.3, seed: 42, cacheHeadroom: 32, ...options };
    this.dynamicCache = options.dynamicCache === undefined ? "patch" : options.dynamicCache;
    this.quantizedStems = options.quantizedStems ?? PRECISION_STEMS["int8-wide"];
    this.yieldToEventLoop = options.yieldToEventLoop ?? createYielder();
    const perf = (globalThis as { performance?: { now(): number } }).performance;
    this.now = options.now ?? (perf ? () => perf.now() : () => Date.now());
  }

  /** Download the bundle metadata, tokenizer and graphs, and create the sessions. */
  static async load(host: EngineHost, options: EngineOptions = {}): Promise<PocketTTSEngine> {
    const engine = new PocketTTSEngine(host, options);
    await engine.init();
    return engine;
  }

  get sampleRate(): number {
    return Number(this.bundle.sample_rate);
  }

  get info(): BundleInfo {
    return {
      language: this.bundle.language,
      sampleRate: this.sampleRate,
      samplesPerFrame: Number(this.bundle.samples_per_frame),
      predefinedVoices: this.bundle.predefined_voices ?? [],
      loadedModelFiles: [...this.loadedModelFiles],
      dynamicCache: this.dynamicCache,
    };
  }

  get isGenerating(): boolean {
    return this.generating;
  }

  private status(s: EngineStatus) {
    this.options.onStatus?.(s);
  }

  private tensor(type: TensorType, data: TensorData, dims: readonly number[]): TensorLike {
    return this.host.tensor(type, data, dims);
  }

  private filename(stem: ModelStem): string {
    return modelFilename(stem, this.quantizedStems, this.dynamicCache === "prebuilt");
  }

  private loadModel(stem: ModelStem): Promise<InferenceSessionLike> {
    const filename = this.filename(stem);
    this.status({ status: "loading-model", stem, filename });
    this.loadedModelFiles.push(filename);
    const patch = this.dynamicCache === "patch" && DYNAMIC_CACHE_STEMS.includes(stem);
    return this.host.loadModel({
      stem,
      filename,
      transform: patch
        ? (bytes) => makeCacheDynamic(bytes, dynamicCacheRules(stem as "flow_lm_main" | "mimi_decoder", this.bundle)).bytes
        : undefined,
    });
  }

  private async init(): Promise<void> {
    this.status({ status: "loading-bundle" });
    this.bundle = JSON.parse(utf8Decode(await this.host.readAsset("bundle.json", "bundle"))) as BundleMetadata;
    this.tokenizer = SentencePieceTokenizer.fromBytes(await this.host.readAsset(this.bundle.tokenizer_file, "tokenizer"));

    // Sessions are created one after another; the next graph downloads while
    // the current one is optimized, so at most two graphs' bytes are in memory.
    const stems = requiredModelStems(Boolean(this.options.voiceCloning));
    for (let i = 0; i < stems.length; i++) {
      if (i + 1 < stems.length) this.host.prefetch?.(this.filename(stems[i + 1]), stems[i + 1]);
      this.sessions[stems[i]] = await this.loadModel(stems[i]);
    }

    this.bos = null;
    if (this.bundle.bos_before_voice_file) {
      try {
        this.bos = parseNpyFloat32(await this.host.readAsset(this.bundle.bos_before_voice_file, "bos"));
      } catch {
        /* optional unless the bundle requires it (checked below) */
      }
    }
    requireGermanBos(this.bundle, this.bos);
    this.flowCacheNames = this.bundle.flow_lm_state_manifest.filter((e) => e.key === "cache").map((e) => e.input_name);
    const step = this.bundle.flow_lm_state_manifest.find((e) => e.key === "step");
    if (!step) throw new Error("bundle.json flow_lm_state_manifest has no step entry");
    this.flowStepName = step.input_name;
    this.status({ status: "ready" });
  }

  // ----- state helpers -----

  private initState(manifest: StateEntry[], shapeFor: (e: StateEntry) => number[] = (e) => e.shape): State {
    const state: State = {};
    for (const entry of manifest) {
      const shape = shapeFor(entry);
      state[entry.input_name] = this.tensor(entry.dtype, makeFilledArray(shape, entry.dtype, entry.fill), shape);
    }
    return state;
  }

  private flowShape(slots: number) {
    return (entry: StateEntry) =>
      this.dynamicCache && entry.key === "cache" ? withAxis(entry.shape, FLOW_CACHE_AXIS, slots) : entry.shape;
  }

  // The decoder graph passes the encoder transformer's caches straight through
  // (Identity), so with dynamic caches they are empty instead of 8 MB per call.
  private mimiShape = (entry: StateEntry) =>
    this.dynamicCache && entry.key === "cache" && entry.module.startsWith("encoder_transformer.")
      ? withAxis(entry.shape, MIMI_CACHE_AXIS, 0)
      : entry.shape;

  private flowStep(state: State): number {
    return Number((state[this.flowStepName].data as BigInt64Array)[0]);
  }

  /** Copy of `state` whose caches hold `slots` positions (filled prefix preserved). */
  private withFlowCapacity(state: State, slots: number): State {
    const used = this.flowStep(state);
    const next = { ...state };
    for (const name of this.flowCacheNames) {
      const cache = state[name];
      next[name] = this.tensor(
        "float32",
        resizeCacheData(cache.data as Float32Array, cache.dims, FLOW_CACHE_AXIS, used, slots),
        withAxis(cache.dims, FLOW_CACHE_AXIS, slots),
      );
    }
    return next;
  }

  /** Grow the caches (rarely, with headroom) before a run that appends `incoming` positions. */
  private ensureFlowCapacity(state: State, incoming: number): State {
    if (!this.dynamicCache) return state;
    const needed = this.flowStep(state) + incoming;
    if (needed <= state[this.flowCacheNames[0]].dims[FLOW_CACHE_AXIS]) return state;
    return this.withFlowCapacity(state, cacheCapacityFor(needed, this.options.cacheHeadroom));
  }

  private static updateState(state: State, result: Record<string, TensorLike>, manifest: StateEntry[]) {
    for (const entry of manifest) state[entry.input_name] = result[entry.output_name];
  }

  private async conditionOnVoice(voice: VoiceEmbedding): Promise<State> {
    let data = voice.data;
    let dims = voice.shape.slice();
    if (this.bundle.insert_bos_before_voice && this.bos) {
      const combined = new Float32Array(this.bos.data.length + data.length);
      combined.set(this.bos.data, 0);
      combined.set(data, this.bos.data.length);
      data = combined;
      dims = [1, dims[1] + this.bos.shape[1], dims[2]];
    }
    const latentDim = Number(this.bundle.latent_dim);
    const state = this.initState(this.bundle.flow_lm_state_manifest, this.flowShape(dims[1]));
    const result = await this.sessions.flow_lm_main!.run({
      sequence: this.tensor("float32", new Float32Array(0), [1, 0, latentDim]),
      text_embeddings: this.tensor("float32", data, dims),
      ...state,
    });
    PocketTTSEngine.updateState(state, result, this.bundle.flow_lm_state_manifest);
    return state;
  }

  // ----- voices -----

  private async encoder(): Promise<InferenceSessionLike> {
    if (this.sessions.mimi_encoder) return this.sessions.mimi_encoder;
    this.encoderLoading ??= this.loadModel("mimi_encoder").catch((err) => {
      this.encoderLoading = null;
      throw err;
    });
    this.sessions.mimi_encoder = await this.encoderLoading;
    return this.sessions.mimi_encoder;
  }

  /**
   * Encode mono PCM at the model sample rate into a voice embedding. The
   * encoder graph is FP32 and shared by every quantization variant, so the
   * result can be stored and reused with loadVoiceEmbedding().
   */
  async encodeVoice(audio: Float32Array): Promise<VoiceEmbedding> {
    const encoder = await this.encoder();
    const outputs = await encoder.run({ audio: this.tensor("float32", audio, [1, 1, audio.length]) });
    const embeddings = outputs[encoder.outputNames[0]];
    let dims = embeddings.dims.slice();
    const data = new Float32Array(embeddings.data as Float32Array);
    while (dims.length > 3 && dims[0] === 1) dims = dims.slice(1);
    if (dims.length < 3) dims = [1, dims[0], dims[1]];
    return { data, shape: dims };
  }

  /** Prepare a voice from an embedding; only the cheap FlowLM conditioning runs. */
  async loadVoiceEmbedding(embedding: VoiceEmbedding, ref: string): Promise<string> {
    validateEmbedding(embedding, Number(this.bundle.conditioning_dim));
    this.voiceStates.set(ref, await this.conditionOnVoice(embedding));
    return ref;
  }

  async cloneVoice(audio: Float32Array, ref: string): Promise<string> {
    return this.loadVoiceEmbedding(await this.encodeVoice(audio), ref);
  }

  /**
   * Prepare a stock voice from a `.npy` embedding asset (float32 [1, frames, dim]).
   * Embeddings come from the FP32 encoder, so one file serves every precision.
   */
  async loadEmbeddingAsset(file: string, ref: string): Promise<string> {
    if (this.voiceStates.has(ref)) return ref;
    const { data, shape } = parseNpyFloat32(await this.host.readAsset(file, "voice"));
    return this.loadVoiceEmbedding({ data, shape }, ref);
  }

  /** Forget a prepared voice. */
  unloadVoice(ref: string): boolean {
    return this.voiceStates.delete(ref);
  }

  hasVoice(ref: string): boolean {
    return this.voiceStates.has(ref);
  }

  // ----- generation -----

  /** Ask a running generate() to stop after the current frame. */
  stop(): void {
    this.stopRequested = true;
  }

  async generate(text: string, voiceRef: string, opts: GenerateOptions = {}): Promise<GenerationMetrics> {
    const deliveryFrames = opts.deliveryFrames ?? 12;
    const firstChunkFrames = opts.firstChunkFrames ?? 3;
    const maxFrames = opts.maxFrames ?? Infinity;
    for (const [name, value] of [["deliveryFrames", deliveryFrames], ["firstChunkFrames", firstChunkFrames]] as const) {
      if (!Number.isInteger(value) || value < 1 || value > 64) throw new Error(`Unsupported ${name}: ${value} (expected an integer 1-64).`);
    }
    if (this.generating) throw new Error("A generation is already in progress.");
    const baseFlowState = this.voiceStates.get(voiceRef);
    if (!baseFlowState) throw new Error(`Voice '${voiceRef}' is not prepared.`);
    const { chunks } = splitIntoChunks(text, this.tokenizer, this.bundle);
    if (!chunks.length) throw new Error("No text to generate.");

    this.generating = true;
    this.stopRequested = false;
    try {
      return await this.run(chunks, baseFlowState, deliveryFrames, firstChunkFrames, maxFrames, opts.onChunk);
    } finally {
      this.generating = false;
    }
  }

  private async run(
    chunks: string[],
    baseFlowState: State,
    deliveryFrames: number,
    firstChunkFrames: number,
    maxFrames: number,
    onChunk: GenerateOptions["onChunk"],
  ): Promise<GenerationMetrics> {
    const bundle = this.bundle;
    const sampleRate = Number(bundle.sample_rate);
    const samplesPerFrame = Number(bundle.samples_per_frame);
    const latentDim = Number(bundle.latent_dim);
    const conditioningDim = Number(bundle.conditioning_dim);
    const textConditioner = this.sessions.text_conditioner!;
    const flowLmMain = this.sessions.flow_lm_main!;
    const flowLmFlow = this.sessions.flow_lm_flow!;
    const mimiDecoder = this.sessions.mimi_decoder!;
    const random = createSeededRandom(this.options.seed);
    const std = Math.sqrt(this.options.temperature);
    const emptySeq = this.tensor("float32", new Float32Array(0), [1, 0, latentDim]);
    const emptyTextEmb = this.tensor("float32", new Float32Array(0), [1, 0, conditioningDim]);
    const dt = 1.0 / LSD_STEPS;
    const st = Array.from({ length: LSD_STEPS }, (_, step) => ({
      s: this.tensor("float32", new Float32Array([step / LSD_STEPS]), [1, 1]),
      t: this.tensor("float32", new Float32Array([step / LSD_STEPS + dt]), [1, 1]),
    }));
    const running = () => !this.stopRequested && totalFrames < maxFrames;

    let isFirstAudioChunk = true;
    let totalFlowLmTime = 0;
    let totalDecodeTime = 0;
    let totalFrames = 0;
    const generationStart = this.now();
    const since = (t: number) => (t - generationStart) / 1000;
    const textChunks: TextChunkReport[] = [];

    for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {
      if (!running()) break;
      const chunkStart = this.now();
      const framesBeforeChunk = totalFrames;
      let isFirstAudioChunkOfTextChunk = true;
      const tokenIds = this.tokenizer.encodeIds(chunks[chunkIdx]);
      // The end-of-speech tail follows this chunk's length, not the whole input's:
      // a short sentence inside a long text still gets the longer tail.
      const framesAfterEos = framesAfterEosForWordCount(countWords(chunks[chunkIdx]), bundle);

      // Each text chunk runs from the clean voice-conditioned state.
      let flowLmState = this.dynamicCache
        ? this.withFlowCapacity(baseFlowState, cacheCapacityFor(this.flowStep(baseFlowState) + tokenIds.length, this.options.cacheHeadroom))
        : { ...baseFlowState };
      const mimiState = this.initState(bundle.mimi_state_manifest, this.mimiShape);
      const textInput = this.tensor("int64", BigInt64Array.from(tokenIds, (t) => BigInt(t)), [1, tokenIds.length]);

      let textEmb = (await textConditioner.run({ token_ids: textInput }))[textConditioner.outputNames[0]];
      if (textEmb.dims.length === 2) {
        textEmb = this.tensor("float32", new Float32Array(textEmb.data as Float32Array), [1, textEmb.dims[0], textEmb.dims[1]]);
      }
      flowLmState = this.ensureFlowCapacity(flowLmState, textEmb.dims[1]);
      const condResult = await flowLmMain.run({ sequence: emptySeq, text_embeddings: textEmb, ...flowLmState });
      PocketTTSEngine.updateState(flowLmState, condResult, bundle.flow_lm_state_manifest);
      const conditioningSeconds = since(this.now());
      let firstAudioSeconds: number | null = null;
      let capped = false;

      const chunkLatents: Float32Array[] = [];
      let chunkDecodedFrames = 0;
      let currentLatent = this.tensor("float32", new Float32Array(latentDim).fill(NaN), [1, 1, latentDim]);
      let eosStep: number | null = null;
      let chunkEnded = false;
      let chunkGenTimeMs = 0;

      for (let step = 0; step < MAX_FRAMES; step++) {
        if (!running()) break;
        if (step > 0) {
          await this.yieldToEventLoop();
          if (!running()) break;
        }

        flowLmState = this.ensureFlowCapacity(flowLmState, 1);
        const stepStart = this.now();
        const arResult = await flowLmMain.run({ sequence: currentLatent, text_embeddings: emptyTextEmb, ...flowLmState });
        const stepElapsed = this.now() - stepStart;
        chunkGenTimeMs += stepElapsed;
        totalFlowLmTime += stepElapsed;

        const conditioning = arResult.conditioning;
        const eosLogit = (arResult.eos_logit.data as Float32Array)[0];
        if (eosLogit > EOS_THRESHOLD && eosStep == null) eosStep = step;
        const shouldStop = eosStep != null && step >= eosStep + framesAfterEos;

        const latentData = new Float32Array(latentDim);
        for (let i = 0; i < latentDim; i++) {
          let u = 0;
          let v = 0;
          while (u === 0) u = random();
          while (v === 0) v = random();
          latentData[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * std;
        }
        for (let lsd = 0; lsd < LSD_STEPS; lsd++) {
          const flowResult = await flowLmFlow.run({
            c: conditioning,
            s: st[lsd].s,
            t: st[lsd].t,
            x: this.tensor("float32", latentData, [1, latentDim]),
          });
          const flowDir = flowResult.flow_dir.data as Float32Array;
          for (let i = 0; i < latentDim; i++) latentData[i] += flowDir[i] * dt;
        }

        chunkLatents.push(new Float32Array(latentData));
        totalFrames++;
        currentLatent = this.tensor("float32", latentData, [1, 1, latentDim]);
        PocketTTSEngine.updateState(flowLmState, arResult, bundle.flow_lm_state_manifest);

        // A chunk that never signals end-of-speech stops at MAX_FRAMES; its
        // generated frames are still decoded so no produced audio is dropped.
        capped = !shouldStop && step === MAX_FRAMES - 1;
        const pending = chunkLatents.length - chunkDecodedFrames;
        let decodeSize = 0;
        if (shouldStop || capped) decodeSize = pending;
        else if (isFirstAudioChunk && pending >= firstChunkFrames) decodeSize = firstChunkFrames;
        else if (pending >= deliveryFrames) decodeSize = deliveryFrames;

        if (decodeSize > 0) {
          const decodeLatents = new Float32Array(decodeSize * latentDim);
          for (let frame = 0; frame < decodeSize; frame++) {
            decodeLatents.set(chunkLatents[chunkDecodedFrames + frame], frame * latentDim);
          }
          const decoderStart = this.now();
          const decodeResult = await mimiDecoder.run({
            latent: this.tensor("float32", decodeLatents, [1, decodeSize, latentDim]),
            ...mimiState,
          });
          const decoderElapsed = this.now() - decoderStart;
          chunkGenTimeMs += decoderElapsed;
          totalDecodeTime += decoderElapsed;
          PocketTTSEngine.updateState(mimiState, decodeResult, bundle.mimi_state_manifest);
          chunkDecodedFrames += decodeSize;

          const audio = new Float32Array(decodeResult[mimiDecoder.outputNames[0]].data as Float32Array);
          firstAudioSeconds ??= since(this.now());
          onChunk?.(audio, {
            chunkDuration: audio.length / sampleRate,
            genTimeSec: chunkGenTimeMs / 1000,
            isFirst: isFirstAudioChunk,
            isLast: shouldStop && chunkIdx === chunks.length - 1,
            chunkStart: isFirstAudioChunkOfTextChunk,
            isSilence: false,
          });
          isFirstAudioChunk = false;
          isFirstAudioChunkOfTextChunk = false;
          chunkGenTimeMs = 0;
        }

        if (shouldStop) {
          chunkEnded = true;
          break;
        }
      }

      textChunks.push({
        index: chunkIdx,
        text: chunks[chunkIdx],
        tokenCount: tokenIds.length,
        framesAfterEos,
        startSeconds: since(chunkStart),
        conditioningSeconds,
        firstAudioSeconds,
        completionSeconds: since(this.now()),
        generatedFrames: totalFrames - framesBeforeChunk,
        emittedFrames: chunkDecodedFrames,
        endReason: chunkEnded ? "eos" : this.stopRequested ? "cancelled" : totalFrames >= maxFrames ? "frame-budget" : capped ? "frame-cap" : "cancelled",
      });

      if (chunkEnded && !this.stopRequested && chunkIdx < chunks.length - 1) {
        const gapSamples = Math.max(1, Math.floor(CHUNK_GAP_SEC * sampleRate));
        onChunk?.(new Float32Array(gapSamples), {
          chunkDuration: gapSamples / sampleRate,
          isFirst: false,
          isLast: false,
          isSilence: true,
        });
      }
    }

    const audioSeconds = (totalFrames * samplesPerFrame) / sampleRate;
    const genTime = (totalFlowLmTime + totalDecodeTime) / 1000;
    return {
      rtfx: genTime > 0 ? audioSeconds / genTime : 0,
      genTime,
      audioDuration: audioSeconds,
      stopped: this.stopRequested || totalFrames >= maxFrames,
      flowLmSeconds: totalFlowLmTime / 1000,
      decoderSeconds: totalDecodeTime / 1000,
      frames: totalFrames,
      deliveryFrames,
      complete: textChunks.length === chunks.length && textChunks.every((c) => c.endReason === "eos"),
      textChunks,
    };
  }
}
