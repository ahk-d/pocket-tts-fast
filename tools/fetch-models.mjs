// Download the pinned model assets for a language from Hugging Face and verify
// every file against the manifest (SHA-256 + size). Nothing is bundled in npm.
//
//   node tools/fetch-models.mjs [--language german] [--precision int8-wide|int8-core|fp32|all]
//        [--cloning] [--out models] [--only file1,file2]
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { ROOT, parseArgs } from "./lib.mjs";
import { MANIFESTS, filesFor } from "../dist/index.js";

const args = parseArgs();
const manifest = MANIFESTS[args.language || "german"];
if (!manifest) throw new Error(`Unknown language ${args.language}`);
const outDir = path.join(path.resolve(args.out || path.join(ROOT, "models")), manifest.language);
const precisions = !args.precision || args.precision === "all" ? Object.keys(manifest.precisions) : [args.precision];
let assets = [...new Map(precisions.flatMap((p) => filesFor(manifest, p, { cloning: Boolean(args.cloning) || !args.precision })).map((a) => [a.file, a])).values()];
if (args.only) assets = assets.filter((a) => args.only.split(",").includes(a.file));
fs.mkdirSync(outDir, { recursive: true });

const sha = (file) =>
  new Promise((resolve, reject) => {
    const h = crypto.createHash("sha256");
    fs.createReadStream(file).on("data", (d) => h.update(d)).on("end", () => resolve(h.digest("hex"))).on("error", reject);
  });

for (const a of assets) {
  const target = path.join(outDir, a.file);
  if (fs.existsSync(target) && fs.statSync(target).size === a.bytes && (await sha(target)) === a.sha256) {
    console.log(`ok       ${a.file}`);
    continue;
  }
  const url = `${manifest.modelBaseUrl}/${a.file}`;
  process.stdout.write(`download ${a.file} (${(a.bytes / 1e6).toFixed(1)} MB) ... `);
  let res;
  for (let attempt = 1; ; attempt++) {
    try {
      res = await fetch(url);
      if (res.ok || !(res.status === 429 || res.status >= 500)) break;
    } catch (err) {
      if (attempt >= 4) throw err;
    }
    if (attempt >= 4) break;
    await new Promise((r) => setTimeout(r, 1000 * attempt));
  }
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const tmp = `${target}.partial`;
  const out = fs.createWriteStream(tmp);
  for await (const chunk of res.body) out.write(chunk);
  await new Promise((r) => out.end(r));
  const got = await sha(tmp);
  if (got !== a.sha256 || fs.statSync(tmp).size !== a.bytes) {
    fs.unlinkSync(tmp);
    throw new Error(`${a.file}: integrity check failed (sha256 ${got})`);
  }
  fs.renameSync(tmp, target);
  console.log("verified");
}
console.log(`\n${assets.length} files in ${outDir}. Stock voice: node tools/prepare-stock-voice.mjs`);
