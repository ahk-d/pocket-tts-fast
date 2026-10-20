// Shared Node helpers for the tools and benchmark: load the engine from a
// local model folder through the same host the browser worker uses.
import fs from "node:fs";
import crypto from "node:crypto";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import * as ort from "onnxruntime-web";
import { PocketTTSEngine, createWebHost, MANIFESTS } from "../dist/index.js";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export function parseArgs(argv = process.argv.slice(2)) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith("--")) continue;
    const key = argv[i].slice(2);
    const next = argv[i + 1];
    out[key] = next === undefined || next.startsWith("--") ? true : (i++, next);
  }
  return out;
}

export const sha256File = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");

// Node's fetch has no file: scheme; serve local files for the host.
export async function fileFetch(url) {
  const u = String(url);
  if (!u.startsWith("file:")) return fetch(url);
  const file = fileURLToPath(u);
  if (!fs.existsSync(file)) return new Response(null, { status: 404 });
  const bytes = fs.readFileSync(file);
  return new Response(bytes, { status: 200, headers: { "content-length": String(bytes.length) } });
}

export async function loadEngine({ modelsDir, language = "german", precision = "int8-wide", dynamicCache = "patch", threads = 4, cloning = false, verify = true }) {
  const manifest = MANIFESTS[language];
  if (!manifest) throw new Error(`Unknown language ${language}`);
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.simd = true;
  const base = pathToFileURL(path.join(path.resolve(modelsDir), language)).href;
  // verify=false skips manifest hashes (only for prebuilt *_dyncache graphs, which aren't in the manifest).
  const host = createWebHost({
    ort,
    manifest: verify ? manifest : { ...manifest, assets: [], voices: [] },
    modelBaseUrl: base,
    cache: false,
    fetch: fileFetch,
  });
  const stems = manifest.precisions[precision];
  if (!stems) throw new Error(`Unknown precision ${precision}`);
  const engine = await PocketTTSEngine.load(host, { quantizedStems: stems, dynamicCache, voiceCloning: cloning });
  return { engine, ort, manifest };
}
