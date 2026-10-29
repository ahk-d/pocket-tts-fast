// Install the actual packed tarball (not a workspace link) into each example
// and build it. Verifies the published file list, entry points and that
// bundlers emit the worker.
//   node scripts/test-consumers.mjs [--keep]
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const packageName = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8")).name;
const run = (cmd, args, cwd) => execFileSync(cmd, args, { cwd, stdio: ["ignore", "pipe", "inherit"], encoding: "utf8" });
const work = fs.mkdtempSync(path.join(process.env.CONSUMER_TMP || os.tmpdir(), "pocket-tts-consumers-"));
const packed = JSON.parse(run("npm", ["pack", "--json", "--pack-destination", work], root))[0];
const tarball = path.join(work, packed.filename);
const { GERMAN, ENGLISH } = await import(path.join(root, "dist/index.js"));
const stockVoices = [GERMAN, ENGLISH].flatMap((language) => language.voices);
const voiceFiles = stockVoices.map((voice) => voice.file);
const files = packed.files.map((f) => f.path).sort();
console.log(`packed ${packed.filename}: ${files.length} files, ${(packed.size / 1024).toFixed(0)} kB`);
for (const required of ["dist/index.js", "dist/worker.js", "dist/index.d.ts", "LICENSE", "README.md", "THIRD_PARTY_NOTICES.md", "package.json", ...voiceFiles, "voices/README.md"]) {
  if (!files.includes(required)) throw new Error(`tarball is missing ${required}`);
}
const leaked = files.filter((f) => !voiceFiles.includes(f) && f !== "voices/README.md" && (!/^(dist\/|package\.json$|README\.md$|LICENSE$|THIRD_PARTY_NOTICES\.md$)/.test(f) || /\.(onnx|npy|wav|bin)$/.test(f)));
if (leaked.length) throw new Error(`unexpected files in tarball: ${leaked.join(", ")}`);

const results = [];
for (const name of ["vanilla", "vite-vanilla", "react-vite"]) {
  const dir = path.join(work, name);
  fs.cpSync(path.join(root, "examples", name), dir, { recursive: true, filter: (f) => !f.includes("node_modules") && !f.includes(`${path.sep}dist`) });
  if (!fs.existsSync(path.join(dir, "package.json"))) fs.writeFileSync(path.join(dir, "package.json"), JSON.stringify({ name: `consumer-${name}`, private: true, type: "module" }));
  run("npm", ["install", "--no-audit", "--no-fund", "--no-package-lock", tarball], dir);
  const installed = fs.realpathSync(path.join(dir, "node_modules", packageName));
  for (const voice of stockVoices) {
    const voiceBytes = fs.readFileSync(path.join(installed, voice.file));
    if (voiceBytes.length !== voice.bytes || createHash("sha256").update(voiceBytes).digest("hex") !== voice.sha256) throw new Error(`${name}: packaged stock voice ${voice.file} fails manifest verification`);
  }
  if (installed.startsWith(root)) throw new Error(`${name}: package resolved to the source tree (link), not the tarball`);
  // Follow the consumer README: copy voices/ into the application static root.
  const usesVite = fs.existsSync(path.join(dir, "vite.config.js"));
  const staticRoot = usesVite ? path.join(dir, "public") : dir;
  fs.cpSync(path.join(installed, "voices"), path.join(staticRoot, "voices"), { recursive: true });
  if (usesVite) {
    run("npx", ["vite", "build", "--logLevel", "warn"], dir);
    for (const voice of stockVoices) {
      const publicVoice = fs.readFileSync(path.join(dir, "dist", voice.file));
      if (createHash("sha256").update(publicVoice).digest("hex") !== voice.sha256) throw new Error(`${name}: build lost or changed the public voice asset ${voice.file}`);
    }
    const assets = fs.readdirSync(path.join(dir, "dist/assets"));
    const worker = assets.find((f) => /^worker.*\.js$/.test(f));
    if (!worker) throw new Error(`${name}: build did not emit the worker (${assets.join(", ")})`);
    const workerSrc = fs.readFileSync(path.join(dir, "dist/assets", worker), "utf8");
    if (!workerSrc.includes("ort.wasm.min.mjs")) throw new Error(`${name}: worker lost the runtime import`);
    results.push(`${name}: vite build ok (worker ${worker})`);
  } else {
    // Plain HTML: the relative import must exist in the installed package.
    const html = fs.readFileSync(path.join(dir, "index.html"), "utf8");
    const spec = html.match(/from "(\.\/node_modules\/[^"]+)"/)[1];
    await import(path.join(dir, spec));
    results.push(`${name}: ${spec} imports cleanly (SSR-safe, no side effects)`);
  }
}
console.log(results.join("\n"));
if (!process.argv.includes("--keep")) fs.rmSync(work, { recursive: true, force: true });
else console.log(`kept ${work}`);
