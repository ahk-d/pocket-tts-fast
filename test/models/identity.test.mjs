// Real-model checks (skipped unless models/<language>/ is present; see models/README.md).
//
//   npm run test:models                      # uses ./models
//   POCKET_TTS_MODELS=/path npm run test:models
//
// Fixed (1000-slot) and dynamic caches must give bit-identical PCM on the same
// backend, precision, voice and seed, for short and multi-sentence input, and
// the dynamic run must actually exercise cache growth.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { pathToFileURL } from "node:url";
import * as ort from "onnxruntime-web";
import { PocketTTSEngine, createWebHost, GERMAN, wavFromChunks } from "../../dist/index.js";
import { ROOT, fileFetch } from "../../tools/lib.mjs";

const modelsDir = path.resolve(process.env.POCKET_TTS_MODELS || path.join(ROOT, "models"));
const langDir = path.join(modelsDir, "german");
const voiceFile = path.join(langDir, GERMAN.voices[0].file);
const have = fs.existsSync(path.join(langDir, "bundle.json")) && fs.existsSync(voiceFile);
const skip = have ? false : `models not found in ${langDir} (run: node tools/fetch-models.mjs && npm run stock-voice)`;
const threads = Number(process.env.POCKET_TTS_THREADS || 4);
const precision = process.env.POCKET_TTS_PRECISION || "int8-wide";

ort.env.wasm.numThreads = threads;

async function load(dynamicCache) {
  const capacities = [];
  const base = createWebHost({ ort, manifest: GERMAN, modelBaseUrl: pathToFileURL(langDir).href, cache: false, fetch: fileFetch });
  const host = {
    ...base,
    async loadModel(req) {
      const session = await base.loadModel(req);
      if (req.stem !== "flow_lm_main") return session;
      // Record the cache length FlowLM receives on every run.
      return { outputNames: session.outputNames, run: (feeds) => (capacities.push(feeds.state_0.dims[2]), session.run(feeds)) };
    },
  };
  const engine = await PocketTTSEngine.load(host, { quantizedStems: GERMAN.precisions[precision], dynamicCache });
  return { engine, capacities };
}

async function pcm(engine, text, voice) {
  const hash = crypto.createHash("sha256");
  let samples = 0;
  await engine.generate(text, voice, {
    deliveryFrames: 3,
    onChunk: (a) => {
      samples += a.length;
      hash.update(Buffer.from(a.buffer, a.byteOffset, a.byteLength));
    },
  });
  return { sha: hash.digest("hex"), samples };
}

const PROMPTS = [
  "Guten Morgen.",
  "Wie spät ist es? Um halb acht fährt der Zug, also beeil dich bitte!",
  "Am Samstag fahren wir früh mit dem Zug in eine kleine Stadt am See. Dort möchten wir zuerst über den Markt gehen, regionale Spezialitäten probieren und danach am Ufer spazieren.",
];

test(`fixed and dynamic KV caches give bit-identical PCM (${precision}, ${threads} threads, stock voice)`, { skip, timeout: 600_000 }, async () => {
  const fixed = await load(false);
  const dynamic = await load("patch");
  const vf = await fixed.engine.loadEmbeddingAsset(GERMAN.voices[0].file, "stock:alba");
  const vd = await dynamic.engine.loadEmbeddingAsset(GERMAN.voices[0].file, "stock:alba");
  assert.deepEqual(dynamic.engine.info.loadedModelFiles.filter((f) => f.includes("encoder")), [], "stock voice needs no encoder");
  for (const text of PROMPTS) {
    const a = await pcm(fixed.engine, text, vf);
    const b = await pcm(dynamic.engine, text, vd);
    assert.ok(a.samples > 0);
    assert.equal(b.samples, a.samples, `same length: ${text}`);
    assert.equal(b.sha, a.sha, `same PCM: ${text}`);
    console.log(`  ${a.sha.slice(0, 16)}  ${(a.samples / 24000).toFixed(2)} s  ${text.slice(0, 40)}`);
  }
  assert.ok(fixed.capacities.every((c) => c === 1000));
  const grown = new Set(dynamic.capacities);
  assert.ok(Math.max(...dynamic.capacities) < 1000, "dynamic caches never needed all 1000 slots");
  assert.ok(grown.size > 2, `cache grew during generation (capacities seen: ${[...grown].join(", ")})`);
});

// Optional: compare with PCM hashes recorded from the original browser-quant-v1
// runtime for its reference clip (embed path, int8-wide, 3-frame delivery).
const refF32 = process.env.POCKET_TTS_REFERENCE_F32;
test("matches hashes recorded from the original runtime (optional)", { skip: !have || !refF32 ? "set POCKET_TTS_REFERENCE_F32" : false, timeout: 600_000 }, async () => {
  const expected = {
    "Guten Morgen.": "2be4e0410fb0399b",
    "Heute üben wir gemeinsam ein paar Wörter.": "79d48600eeb3829a",
    "Nach dem Unterricht gehe ich in den Laden an der Ecke, weil ich Brot und Obst für das Abendessen brauche.": "97f41df7ef811f89",
  };
  const { engine } = await load("patch");
  const buf = fs.readFileSync(refF32);
  const audio = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const ref = await engine.loadVoiceEmbedding(await engine.encodeVoice(audio), "reference");
  for (const [text, sha] of Object.entries(expected)) assert.equal((await pcm(engine, text, ref)).sha.slice(0, 16), sha, text);
});

// Endings: each sentence is its own chunk with its own EOS tail
// (src/core/text.ts). Every chunk must end by end-of-speech, deliver all of its
// frames, and every sentence must decay toward silence: a clipped last sound
// leaves the final 100 ms loud. Calibrated with Whisper word timings: the pre-fix
// "two-sentences" case measured RMS 0.051 (clipped); complete endings on vowels
// ("Danke.", "brauche.") measure up to ~0.03. Transcripts: tools/check_transcripts.py.
const ENDINGS = JSON.parse(fs.readFileSync(new URL("./endings.de.json", import.meta.url), "utf8"));
const WAV_DIR = process.env.POCKET_TTS_WAV_DIR;

test("utterance endings decay to near-silence and every sentence ends by EOS", { skip, timeout: 900_000 }, async () => {
  const { engine } = await load("patch");
  const voice = await engine.loadEmbeddingAsset(GERMAN.voices[0].file, "stock:alba");
  const problems = [];
  for (const { id, text } of ENDINGS) {
    const chunks = [];
    const metrics = await engine.generate(text, voice, { deliveryFrames: 3, onChunk: (a, meta) => chunks.push([a, meta]) });
    const all = Float32Array.from(chunks.flatMap(([a]) => [...a]));
    const rms = (from, to) => Math.sqrt(all.subarray(from, to).reduce((s, v) => s + v * v, 0) / Math.max(1, to - from));
    const tail = rms(all.length - 2400, all.length);
    // Each sentence's own ending: the 100 ms before every inserted pause.
    let offset = 0;
    const sentenceTails = [];
    for (const [a, meta] of chunks) {
      if (meta.isSilence) sentenceTails.push(rms(offset - 2400, offset));
      offset += a.length;
    }
    sentenceTails.push(tail);
    if (WAV_DIR) {
      fs.mkdirSync(WAV_DIR, { recursive: true });
      fs.writeFileSync(path.join(WAV_DIR, `${id}.wav`), Buffer.from(wavFromChunks(chunks.map(([a]) => a), 24000).buffer));
    }
    console.log(`  ${id.padEnd(26)} ${(all.length / 24000).toFixed(2)} s  chunks ${metrics.textChunks.length}  sentence-end RMS ${sentenceTails.map((r) => r.toFixed(4)).join(" ")}`);
    // Runaway generation (e.g. the model babbling on unknown byte tokens) shows up as far too much audio.
    const seconds = all.length / 24000;
    if (seconds > 1 + 0.12 * text.length) problems.push(`${id}: ${seconds.toFixed(1)} s is implausibly long for ${text.length} characters`);
    if (!metrics.complete) problems.push(`${id}: incomplete (${metrics.textChunks.map((c) => c.endReason).join(",")})`);
    for (const c of metrics.textChunks) if (c.emittedFrames !== c.generatedFrames) problems.push(`${id}: chunk ${c.index} delivered ${c.emittedFrames}/${c.generatedFrames} frames`);
    sentenceTails.forEach((r, k) => r > 0.04 && problems.push(`${id}: sentence ${k + 1} ends loud (last 100 ms RMS ${r.toFixed(3)})`));
  }
  assert.deepEqual(problems, []);
});
