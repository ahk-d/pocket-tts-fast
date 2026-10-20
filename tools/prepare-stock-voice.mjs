// Build a stock voice embedding (.npy) from a licensed reference clip.
//
//   node tools/prepare-stock-voice.mjs [--voice alba] [--source path/to/casual.wav]
//        [--models models] [--language german]
//
// The clip is downloaded from its pinned URL (or read from --source) and must
// match the manifest SHA-256. It is resampled to 24 kHz (linear), cut to 10 s
// and encoded with the FP32 mimi_encoder. The resulting embedding is
// precision-independent: every precision profile conditions on the same file.
// Output: models/<language>/voices/<id>.emb.npy plus its hash and size, which
// belong in src/manifest.ts and docs/release-assets.json.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { ROOT, loadEngine, parseArgs } from "./lib.mjs";
import { MANIFESTS, decodeWav, encodeNpyFloat32 } from "../dist/index.js";
import { prepareReferenceAudio } from "../dist/index.js";

const args = parseArgs();
const language = args.language || "german";
const modelsDir = path.resolve(args.models || path.join(ROOT, "models"));
const manifest = MANIFESTS[language];
const voice = manifest.voices.find((v) => v.id === (args.voice || "alba"));
if (!voice) throw new Error(`Unknown voice ${args.voice}`);

let wav;
if (args.source) wav = fs.readFileSync(args.source);
else {
  console.log(`Downloading ${voice.source.url}`);
  const res = await fetch(voice.source.url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  wav = Buffer.from(await res.arrayBuffer());
}
const sourceSha = crypto.createHash("sha256").update(wav).digest("hex");
if (sourceSha !== voice.source.sha256) throw new Error(`Source clip SHA-256 ${sourceSha} does not match the manifest (${voice.source.sha256})`);

const { audio, sampleRate } = decodeWav(wav);
const { engine } = await loadEngine({ modelsDir, language, cloning: true });
const pcm = prepareReferenceAudio(audio, { inputSampleRate: sampleRate, targetRate: engine.sampleRate, maxSeconds: 10 });
const embedding = await engine.encodeVoice(pcm);
const npy = encodeNpyFloat32(embedding.data, embedding.shape);
const out = path.join(modelsDir, language, voice.file);
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, npy);
const record = {
  id: voice.id,
  file: voice.file,
  sha256: crypto.createHash("sha256").update(npy).digest("hex"),
  bytes: npy.length,
  shape: embedding.shape,
  embeddingDataSha256: crypto.createHash("sha256").update(Buffer.from(embedding.data.buffer)).digest("hex"),
  preparedAudio: { sampleRate: engine.sampleRate, samples: pcm.length, sha256: crypto.createHash("sha256").update(Buffer.from(pcm.buffer, pcm.byteOffset, pcm.byteLength)).digest("hex") },
  source: { ...voice.source, sampleRate, frames: audio.length },
  encoder: manifest.assets.find((a) => a.file === "mimi_encoder.onnx"),
  modelRevision: manifest.revision,
};
console.log(JSON.stringify(record, null, 2));
process.exit(0);
