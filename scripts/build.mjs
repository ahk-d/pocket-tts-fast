// Bundle the client and the worker into two self-contained ES modules so
// dist/ works from a plain <script type="module"> and in any bundler that
// understands `new Worker(new URL("./worker.js", import.meta.url))`.
// onnxruntime-web is not bundled: the worker imports a pinned version at runtime.
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { build } from "esbuild";

// Content hash of the sources, baked into both bundles. The worker reports it
// on init and the client refuses a different one, so a stale cached worker.js
// can never run behind a newer client (or the reverse).
const hash = createHash("sha256");
const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true })
    .sort((a, b) => a.name.localeCompare(b.name))
    .forEach((e) => (e.isDirectory() ? walk(join(dir, e.name)) : hash.update(e.name).update(readFileSync(join(dir, e.name)))));
walk("src");
const pkg = JSON.parse(readFileSync("package.json", "utf8"));
const BUILD_ID = `${pkg.version}+${hash.digest("hex").slice(0, 12)}`;

rmSync("dist", { recursive: true, force: true });
await build({
  entryPoints: { index: "src/index.ts", worker: "src/worker.ts" },
  outdir: "dist",
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  sourcemap: false,
  legalComments: "inline",
  logLevel: "warning",
  define: { __POCKET_TTS_BUILD__: JSON.stringify(BUILD_ID) },
});
console.log(`build ${BUILD_ID}`);
