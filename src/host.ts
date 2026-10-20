// EngineHost for onnxruntime-web. Assets listed in the manifest are fetched
// from a pinned revision, checked against their SHA-256 and byte size, and kept
// in Cache Storage under a revision-keyed cache so later visits skip the
// download and an old revision can never be mixed with a new one.

import type { EngineHost, InferenceSessionLike, TensorData, TensorType } from "./core/index.js";
import { assetFor, type LanguageManifest } from "./manifest.js";

/** The parts of the onnxruntime-web module this host uses. */
export interface OrtWebModule {
  Tensor: new (type: TensorType, data: TensorData, dims: readonly number[]) => { dims: readonly number[]; data: unknown };
  InferenceSession: {
    create(model: Uint8Array, options?: Record<string, unknown>): Promise<InferenceSessionLike>;
  };
  env: { wasm: { numThreads?: number; simd?: boolean; wasmPaths?: string | Record<string, string> }; versions?: Record<string, string> };
}

export interface ProgressEvent {
  label: string;
  loaded: number;
  total: number;
  fromCache: boolean;
}

export interface WebHostOptions {
  ort: OrtWebModule;
  manifest: LanguageManifest;
  /** Folder with the graphs, bundle.json and tokenizer. Default: the manifest's pinned Hugging Face URL. */
  modelBaseUrl?: string;
  /** Folder with stock voice files (`voices/...`). Default: modelBaseUrl. */
  voiceBaseUrl?: string;
  /** Persist downloads in Cache Storage (needs a secure context). Default true. */
  cache?: boolean;
  /** Cache name prefix; the manifest id is appended. */
  cachePrefix?: string;
  onProgress?: (event: ProgressEvent) => void;
  fetch?: typeof fetch;
  /** Attempts per asset for network errors, 429 and 5xx. Default 3. */
  attempts?: number;
  /** Wait between attempts (ms), multiplied by the attempt number unless Retry-After says otherwise. */
  retryDelayMs?: number;
  sessionOptions?: Record<string, unknown>;
}

export const CACHE_PREFIX = "pocket-tts:";

export class AssetIntegrityError extends Error {}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes as unknown as BufferSource);
  let hex = "";
  for (const b of new Uint8Array(digest)) hex += b.toString(16).padStart(2, "0");
  return hex;
}

async function readBody(response: Response, label: string, fromCache: boolean, onProgress?: (e: ProgressEvent) => void): Promise<Uint8Array> {
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) {
    const buf = new Uint8Array(await response.arrayBuffer());
    onProgress?.({ label, loaded: buf.byteLength, total: buf.byteLength, fromCache });
    return buf;
  }
  // Read straight into one buffer of the declared size.
  const reader = response.body.getReader();
  let out = new Uint8Array(total);
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > out.length) {
      const grown = new Uint8Array(Math.max(out.length * 2, loaded + value.length));
      grown.set(out.subarray(0, loaded));
      out = grown;
    }
    out.set(value, loaded);
    loaded += value.length;
    onProgress?.({ label, loaded, total, fromCache });
  }
  if (loaded !== total) throw new Error(`Incomplete download of ${label}: ${loaded} of ${total} bytes`);
  return out;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Revision-keyed Cache Storage name for a manifest. */
export function cacheNameFor(manifest: LanguageManifest, prefix = CACHE_PREFIX): string {
  return `${prefix}${manifest.id}`;
}

/** Delete caches of other revisions of the same language. Returns the names deleted. */
export async function pruneOldCaches(manifest: LanguageManifest, prefix = CACHE_PREFIX): Promise<string[]> {
  if (typeof caches === "undefined") return [];
  const current = cacheNameFor(manifest, prefix);
  const sameLanguage = `${prefix}${manifest.language}@`;
  const deleted: string[] = [];
  for (const name of await caches.keys()) {
    if (name.startsWith(sameLanguage) && name !== current && (await caches.delete(name))) deleted.push(name);
  }
  return deleted;
}

export function createWebHost(options: WebHostOptions): EngineHost & { prefetch(filename: string, label: string): void } {
  const { ort, manifest, onProgress } = options;
  const doFetch = options.fetch ?? ((input: RequestInfo | URL, init?: RequestInit) => fetch(input, init));
  const attempts = Math.max(1, options.attempts ?? 3);
  const retryDelayMs = options.retryDelayMs ?? 500;
  const modelBase = (options.modelBaseUrl ?? manifest.modelBaseUrl).replace(/\/$/, "");
  const voiceBase = (options.voiceBaseUrl ?? modelBase).replace(/\/$/, "");
  const voiceFiles = new Map(manifest.voices.map((v) => [v.file, v]));
  const urlFor = (file: string) =>
    /^[a-z][a-z0-9+.-]*:/i.test(file) || file.startsWith("/") ? file : `${voiceFiles.has(file) ? voiceBase : modelBase}/${file}`;
  const expected = (file: string): { sha256: string; bytes: number } | null => {
    const a = assetFor(manifest, file) ?? voiceFiles.get(file);
    return a && a.sha256 ? { sha256: a.sha256, bytes: a.bytes } : null;
  };
  const inFlight = new Map<string, Promise<Uint8Array>>();
  let cachePromise: Promise<Cache | null> | null = null;

  const openCache = () =>
    (cachePromise ??= (async () => {
      if (options.cache === false || typeof caches === "undefined") return null;
      try {
        return await caches.open(cacheNameFor(manifest, options.cachePrefix));
      } catch {
        return null;
      }
    })());

  async function verify(file: string, bytes: Uint8Array) {
    const want = expected(file);
    if (!want) return;
    if (bytes.byteLength !== want.bytes) throw new AssetIntegrityError(`${file}: expected ${want.bytes} bytes, got ${bytes.byteLength}`);
    const got = await sha256Hex(bytes);
    if (got !== want.sha256) throw new AssetIntegrityError(`${file}: SHA-256 mismatch (expected ${want.sha256}, got ${got})`);
  }

  async function fromNetwork(url: string, file: string, label: string): Promise<Uint8Array> {
    let lastError: unknown = null;
    for (let attempt = 1; attempt <= attempts; attempt++) {
      let res: Response;
      try {
        res = await doFetch(url);
      } catch (err) {
        lastError = err;
        if (attempt < attempts) await sleep(retryDelayMs * attempt);
        continue;
      }
      if (res.ok) {
        try {
          const bytes = await readBody(res, label, false, onProgress);
          await verify(file, bytes);
          return bytes;
        } catch (err) {
          // An integrity failure is not transient: never retry into a different file.
          if (err instanceof AssetIntegrityError) throw err;
          lastError = err;
          if (attempt < attempts) await sleep(retryDelayMs * attempt);
          continue;
        }
      }
      lastError = new Error(`Failed to fetch ${label} (${url}): HTTP ${res.status}`);
      if (!(res.status === 429 || res.status >= 500) || attempt === attempts) throw lastError;
      const retryAfter = Number(res.headers.get("retry-after"));
      await sleep(Number.isFinite(retryAfter) && retryAfter > 0 ? Math.min(retryAfter, 30) * 1000 : retryDelayMs * attempt);
    }
    throw lastError instanceof Error ? lastError : new Error(`Failed to fetch ${label}`);
  }

  async function download(file: string, label: string): Promise<Uint8Array> {
    const url = urlFor(file);
    const cache = await openCache();
    if (cache) {
      try {
        const hit = await cache.match(url);
        if (hit) {
          const bytes = await readBody(hit, label, true, onProgress);
          // Entries are only written after verification; the size check catches truncated writes.
          const want = expected(file);
          if (!want || bytes.byteLength === want.bytes) return bytes;
          await cache.delete(url);
        }
      } catch {
        /* fall through to the network */
      }
    }
    const bytes = await fromNetwork(url, file, label);
    if (cache) {
      const headers = { "content-length": String(bytes.byteLength), "content-type": "application/octet-stream" };
      // Don't wait for the disk write; a full quota only means no caching.
      cache.put(url, new Response(bytes as unknown as BodyInit, { headers })).catch(() => {});
    }
    return bytes;
  }

  // A prefetched download is held until its consumer takes it.
  const prefetched = new Map<string, Promise<Uint8Array>>();

  // One request per asset even if several callers ask at once.
  function fetchOnce(file: string, label: string): Promise<Uint8Array> {
    const url = urlFor(file);
    const early = prefetched.get(url);
    if (early) {
      prefetched.delete(url);
      return early;
    }
    let p = inFlight.get(url);
    if (!p) {
      const created = download(file, label);
      inFlight.set(url, created);
      created
        .catch(() => {})
        .finally(() => {
          if (inFlight.get(url) === created) inFlight.delete(url);
        });
      p = created;
    }
    return p;
  }

  return {
    tensor: (type, data, dims) => new ort.Tensor(type, data, dims),
    readAsset: (filename, label) => fetchOnce(filename, label),
    prefetch(filename, label) {
      const url = urlFor(filename);
      if (prefetched.has(url)) return;
      const p = fetchOnce(filename, label);
      p.catch(() => {}); // surfaced when awaited by loadModel
      prefetched.set(url, p);
    },
    async loadModel({ stem, filename, transform }) {
      let bytes = await fetchOnce(filename, stem);
      if (transform) bytes = transform(bytes);
      return ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
        ...options.sessionOptions,
      });
    },
  };
}
