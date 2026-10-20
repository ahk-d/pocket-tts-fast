// Headless benchmark: the same engine and asset host the browser worker runs,
// on the onnxruntime-web WASM build inside Node. Adapted from
// browser-quant-v1/bench_worker.mjs. See bench/protocol.md.
//
//   node bench/bench.mjs [--models models] [--precision int8-wide] [--threads 4]
//        [--cache dynamic|fixed|both] [--voice alba | --reference clip.wav|.f32]
//        [--frames 3] [--prompts all|id,...] [--repeat 5] [--json out.json]
//        [--allow-load]   # run even if the machine is busy (labelled "stress")
//
// Per run: wall time, audio length, RTF (wall / audio), first-chunk time,
// simulated live-playback gaps (chunks played the moment they arrive), KV-state
// bytes passed into FlowLM / the decoder per call, and the PCM SHA-256. With
// --cache both, fixed and dynamic runs alternate and must produce identical PCM.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { pathToFileURL } from "node:url";
import * as ort from "onnxruntime-web";
import { PocketTTSEngine, createWebHost, MANIFESTS, decodeWav, prepareReferenceAudio, simulateLivePlayback, median } from "../dist/index.js";
import { ROOT, fileFetch, parseArgs } from "../tools/lib.mjs";

const args = parseArgs();
const language = args.language || "german";
const manifest = MANIFESTS[language];
const modelsDir = path.resolve(args.models || path.join(ROOT, "models"));
const precision = args.precision || "int8-wide";
const threads = Number(args.threads || 4);
const deliveryFrames = Number(args.frames || 3);
const repeat = Number(args.repeat || 5);
const cacheModes = { dynamic: ["dynamic"], fixed: ["fixed"], both: ["fixed", "dynamic"] }[args.cache || "both"];
if (!manifest.precisions[precision]) throw new Error(`--precision must be one of ${Object.keys(manifest.precisions)}`);
if (!cacheModes) throw new Error("--cache must be dynamic, fixed or both");

const load1 = os.loadavg()[0];
const busy = load1 > os.cpus().length / 2;
if (busy && !args["allow-load"]) {
  console.error(`Load average ${load1.toFixed(1)} on ${os.cpus().length} cores: timings would not be representative. Close other work, or pass --allow-load to record a labelled stress run.`);
  process.exit(2);
}

let prompts = JSON.parse(fs.readFileSync(path.join(ROOT, "bench/prompts.de.json"), "utf8"));
if (args.prompts && args.prompts !== "all") prompts = prompts.filter((p) => String(args.prompts).split(",").some((s) => p.id.includes(s)));

ort.env.wasm.numThreads = threads;
ort.env.wasm.simd = true;

async function loadEngine(mode) {
  const stats = { flowCalls: 0, flowStateBytes: 0, decoderCalls: 0, decoderStateBytes: 0 };
  const stateBytes = (feeds, names) => names.reduce((n, k) => n + (feeds[k]?.data?.byteLength ?? 0), 0);
  const base = createWebHost({ ort, manifest, modelBaseUrl: pathToFileURL(path.join(modelsDir, language)).href, cache: false, fetch: fileFetch });
  let engine;
  const host = {
    ...base,
    async loadModel(req) {
      const session = await base.loadModel(req);
      if (req.stem !== "flow_lm_main" && req.stem !== "mimi_decoder") return session;
      return {
        outputNames: session.outputNames,
        run(feeds) {
          const manifestKey = req.stem === "flow_lm_main" ? "flow_lm_state_manifest" : "mimi_state_manifest";
          const names = engine.bundle[manifestKey].map((e) => e.input_name);
          if (req.stem === "flow_lm_main") (stats.flowCalls++, (stats.flowStateBytes += stateBytes(feeds, names)));
          else (stats.decoderCalls++, (stats.decoderStateBytes += stateBytes(feeds, names)));
          return session.run(feeds);
        },
      };
    },
  };
  const t0 = performance.now();
  engine = await PocketTTSEngine.load(host, { quantizedStems: manifest.precisions[precision], dynamicCache: mode === "dynamic" ? "patch" : false });
  return { engine, stats, loadSeconds: (performance.now() - t0) / 1000 };
}

async function prepareVoice(engine) {
  if (!args.reference) return engine.loadEmbeddingAsset(manifest.voices.find((v) => v.id === (args.voice || "alba")).file, "voice");
  const bytes = fs.readFileSync(args.reference);
  const audio = args.reference.endsWith(".f32")
    ? new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
    : (({ audio, sampleRate }) => prepareReferenceAudio(audio, { inputSampleRate: sampleRate, targetRate: engine.sampleRate }))(decodeWav(bytes));
  return engine.loadVoiceEmbedding(await engine.encodeVoice(audio), "voice");
}

async function run(setup, prompt) {
  const { engine, stats, voice } = setup;
  const before = { ...stats };
  const chunks = [];
  const arrivals = [];
  const start = performance.now();
  const metrics = await engine.generate(prompt.text, voice, {
    deliveryFrames,
    onChunk: (audio) => {
      chunks.push(audio);
      arrivals.push((performance.now() - start) / 1000);
    },
  });
  const wall = (performance.now() - start) / 1000;
  const samples = chunks.reduce((n, c) => n + c.length, 0);
  const audio = samples / engine.sampleRate;
  const hash = crypto.createHash("sha256");
  for (const c of chunks) hash.update(Buffer.from(c.buffer, c.byteOffset, c.byteLength));
  const flowCalls = stats.flowCalls - before.flowCalls;
  const decoderCalls = stats.decoderCalls - before.decoderCalls;
  return {
    id: prompt.id,
    wall,
    audio,
    rtf: wall / audio,
    first: arrivals[0],
    ...simulateLivePlayback(chunks.map((c, i) => ({ seconds: c.length / engine.sampleRate, arrival: arrivals[i] }))),
    frames: metrics.frames,
    flowLmSeconds: metrics.flowLmSeconds,
    decoderSeconds: metrics.decoderSeconds,
    flowStateMBPerCall: (stats.flowStateBytes - before.flowStateBytes) / flowCalls / 1e6,
    decoderStateMBPerCall: (stats.decoderStateBytes - before.decoderStateBytes) / decoderCalls / 1e6,
    sha256: hash.digest("hex"),
  };
}

const environment = {
  cpu: os.cpus()[0]?.model,
  logicalCores: os.cpus().length,
  platform: `${os.platform()} ${os.release()}`,
  node: process.version,
  onnxruntimeWeb: ort.env.versions?.web,
  threads,
  loadAverageAtStart: os.loadavg(),
  label: busy ? "stress (machine busy)" : "controlled",
};
console.log(`${environment.cpu} · ${environment.logicalCores} logical cores · Node ${environment.node} · onnxruntime-web ${environment.onnxruntimeWeb} · load ${load1.toFixed(1)} (${environment.label})`);
console.log(`${manifest.id} · ${precision} · ${threads} threads · ${deliveryFrames}-frame delivery · voice ${args.reference ? path.basename(args.reference) : args.voice || "alba"} · ${repeat}× rotated`);

const setups = {};
for (const mode of cacheModes) {
  const s = await loadEngine(mode);
  s.voice = await prepareVoice(s.engine);
  await run(s, { id: "warmup", text: prompts[0].text }); // one complete warmup per configuration
  setups[mode] = s;
  console.log(`${mode.padEnd(7)} load ${s.loadSeconds.toFixed(2)} s`);
}

const rows = [];
for (let r = 0; r < repeat; r++) {
  const order = prompts.map((_, i) => prompts[(i + r) % prompts.length]); // rotate order each repetition
  for (const p of order) {
    for (const mode of r % 2 ? [...cacheModes].reverse() : cacheModes) {
      const row = { repetition: r, cache: mode, ...(await run(setups[mode], p)) };
      rows.push(row);
      console.log(
        `${mode.padEnd(7)} ${row.id.padEnd(17)} wall ${row.wall.toFixed(2)}s audio ${row.audio.toFixed(2)}s RTF ${row.rtf.toFixed(3)} first ${row.first.toFixed(2)}s gaps ${row.gaps} (${row.gapSeconds.toFixed(2)}s) state/call flow ${row.flowStateMBPerCall.toFixed(2)} MB dec ${row.decoderStateMBPerCall.toFixed(2)} MB pcm ${row.sha256.slice(0, 12)}`,
      );
    }
  }
}

console.log("\nmedians (range) per prompt");
const mismatches = [];
for (const p of prompts) {
  const line = [p.id.padEnd(17)];
  for (const mode of cacheModes) {
    const rs = rows.filter((r) => r.id === p.id && r.cache === mode);
    const rtf = rs.map((r) => r.rtf);
    line.push(`${mode} RTF ${median(rtf).toFixed(3)} (${Math.min(...rtf).toFixed(3)}–${Math.max(...rtf).toFixed(3)}) gen ${median(rs.map((r) => r.wall)).toFixed(2)}s first ${median(rs.map((r) => r.first)).toFixed(2)}s gaps ${median(rs.map((r) => r.gaps))}`);
  }
  if (cacheModes.length === 2) {
    const shas = new Set(rows.filter((r) => r.id === p.id).map((r) => r.sha256));
    if (shas.size !== 1) mismatches.push(p.id);
    const f = median(rows.filter((r) => r.id === p.id && r.cache === "fixed").map((r) => r.wall));
    const d = median(rows.filter((r) => r.id === p.id && r.cache === "dynamic").map((r) => r.wall));
    line.push(`speedup ${(f / d).toFixed(2)}× · PCM ${shas.size === 1 ? "identical" : "DIFFERS"}`);
  }
  console.log(line.join(" · "));
}
if (args.json) {
  fs.mkdirSync(path.dirname(path.resolve(args.json)), { recursive: true });
  fs.writeFileSync(args.json, JSON.stringify({ environment, manifest: manifest.id, precision, deliveryFrames, repeat, prompts, rows }, null, 2));
}
if (mismatches.length) {
  console.error(`PCM differs between fixed and dynamic caches for: ${mismatches.join(", ")}`);
  process.exit(1);
}
process.exit(0);
