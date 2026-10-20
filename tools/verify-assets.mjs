// Check a local model folder against the manifest: every present file must
// match its SHA-256 and size; report missing ones. Exit code 1 on mismatch.
//   node tools/verify-assets.mjs [--models models] [--language german]
import fs from "node:fs";
import path from "node:path";
import { ROOT, parseArgs, sha256File } from "./lib.mjs";
import { MANIFESTS } from "../dist/index.js";

const args = parseArgs();
const manifest = MANIFESTS[args.language || "german"];
const dir = path.join(path.resolve(args.models || path.join(ROOT, "models")), manifest.language);
let bad = 0;
for (const a of [...manifest.assets, ...manifest.voices]) {
  const file = path.join(dir, a.file);
  if (!fs.existsSync(file)) {
    console.log(`missing  ${a.file}`);
    continue;
  }
  const size = fs.statSync(file).size;
  const ok = size === a.bytes && sha256File(file) === a.sha256;
  if (!ok) bad++;
  console.log(`${ok ? "ok      " : "MISMATCH"} ${a.file}`);
}
process.exit(bad ? 1 : 0);
