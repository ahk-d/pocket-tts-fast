// The small slice of the onnxruntime API the engine needs. onnxruntime-web,
// onnxruntime-node and onnxruntime-react-native all satisfy it (they share
// onnxruntime-common), and tests can supply fakes.

export type TensorType = "float32" | "int64" | "bool";
export type TensorData = Float32Array | BigInt64Array | Uint8Array;

export interface TensorLike {
  readonly dims: readonly number[];
  readonly data: unknown;
  readonly type?: string;
}

export interface InferenceSessionLike {
  readonly outputNames: readonly string[];
  run(feeds: Record<string, TensorLike>): Promise<Record<string, TensorLike>>;
}

export type ModelStemName = "mimi_encoder" | "text_conditioner" | "flow_lm_main" | "flow_lm_flow" | "mimi_decoder";

export interface ModelRequest {
  stem: ModelStemName;
  /** File name inside the language folder, e.g. `flow_lm_main_int8.onnx`. */
  filename: string;
  /**
   * When set, the host must pass the model bytes through this function before
   * creating the session (in-memory dynamic-cache patch). A host that can only
   * load models by file path should use `dynamicCache: "prebuilt"` instead.
   */
  transform?: (bytes: Uint8Array) => Uint8Array;
}

/** Platform glue: tensors, sessions and asset bytes. */
export interface EngineHost {
  tensor(type: TensorType, data: TensorData, dims: readonly number[]): TensorLike;
  /** Bytes of a bundle file (bundle.json, tokenizer.model, voices.bin, ...). */
  readAsset(filename: string, label: string): Promise<Uint8Array>;
  /** Create an inference session for one graph. */
  loadModel(request: ModelRequest): Promise<InferenceSessionLike>;
  /** Optional hint: start downloading a graph that loadModel() will ask for next. */
  prefetch?(filename: string, label: string): void;
}

/** Let other tasks (a stop request, UI) run between generation steps. */
export type Yielder = () => Promise<void>;

/**
 * A macrotask without the ~4 ms clamp of nested setTimeout(0): setImmediate
 * (Node, React Native), else a MessageChannel (browsers, workers), else setTimeout.
 */
export function createYielder(): Yielder {
  const g = globalThis as { setImmediate?: (fn: () => void) => unknown; MessageChannel?: typeof MessageChannel };
  if (typeof g.setImmediate === "function") {
    const setImmediate = g.setImmediate;
    return () => new Promise((resolve) => void setImmediate(resolve));
  }
  const MC = g.MessageChannel;
  if (typeof MC !== "function") return () => new Promise((resolve) => setTimeout(resolve, 0));
  let channel: MessageChannel | null = null;
  const waiting: (() => void)[] = [];
  return () =>
    new Promise<void>((resolve) => {
      if (!channel) {
        channel = new MC();
        channel.port1.onmessage = () => waiting.shift()?.();
      }
      waiting.push(resolve);
      channel.port2.postMessage(0);
    });
}
