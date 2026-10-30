// Static site for GitHub Pages, Vercel or any static host: the comparison page
// at the site root, the full demo at demo/, plus the built package, stock voices
// and the pinned onnxruntime-web files. Only the model graphs load from
// elsewhere (the commit-pinned Hugging Face revision). Paths are relative so the
// site also works under a sub-path such as https://<user>.github.io/<repo>/.
//   npm run build:site   → ./site
import { cpSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";

const out = "site";
const rewrite = (file, from, to) => writeFileSync(file, readFileSync(file, "utf8").replaceAll(from, to));
// Static hosts cache files for a while (GitHub Pages: 10 min) with no way to
// purge, so a browser can pair a new index.js with an old worker.js. The package
// lives in a content-hashed folder and the entry scripts carry the same hash,
// so each deploy is fetched as a whole.
const hash = createHash("sha256");
for (const f of readdirSync("dist", { recursive: true }).sort()) if (/\.js$/.test(f)) hash.update(readFileSync(join("dist", f)));
const pkg = `dist-${hash.digest("hex").slice(0, 10)}`;

rmSync(out, { recursive: true, force: true });
mkdirSync(out);
cpSync("demo/compare", out, { recursive: true });
rewrite(join(out, "compare.js"), '"/dist/index.js"', `"./${pkg}/index.js"`);
rewrite(join(out, "index.html"), '"./compare.js"', `"./compare.js?v=${pkg}"`);

mkdirSync(join(out, "demo"));
for (const f of ["index.html", "main.js", "style.css"]) cpSync(join("demo", f), join(out, "demo", f));
rewrite(join(out, "demo", "main.js"), '"/dist/index.js"', `"../${pkg}/index.js"`);
rewrite(join(out, "demo", "index.html"), '"./main.js"', `"./main.js?v=${pkg}"`);
rewrite(join(out, "demo", "index.html"), '<link rel="stylesheet" href="./style.css">', '<link rel="stylesheet" href="./style.css">\n  <script src="../coi-serviceworker.min.js"></script>');

cpSync("dist", join(out, pkg), { recursive: true });
cpSync("voices", join(out, "voices"), { recursive: true });
const ort = "node_modules/onnxruntime-web/dist";
mkdirSync(join(out, "ort"));
for (const f of readdirSync(ort)) {
  if (f === "ort.wasm.min.mjs" || /^ort-wasm-simd-threaded\.(mjs|wasm)$/.test(f)) cpSync(join(ort, f), join(out, "ort", f));
}
writeFileSync(join(out, ".nojekyll"), "");
console.error(`site → ${out}/ (package in ${pkg}/)`);
