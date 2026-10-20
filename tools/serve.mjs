// Local static server for the demo and examples, with the headers WASM threads
// need (cross-origin isolation). Serves:
//   /            demo/
//   /dist/       the built package
//   /ort/        onnxruntime-web dist from node_modules (pinned 1.20.0)
//   /models/     ./models (graphs + stock voices; see models/README.md)
//   /examples/   examples/
//   node tools/serve.mjs [--port 8787] [--no-isolation]
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { ROOT, parseArgs } from "./lib.mjs";

const args = parseArgs();
const port = Number(args.port || 8787);
const isolate = !args["no-isolation"];
const routes = [
  ["/dist/", path.join(ROOT, "dist")],
  ["/ort/", path.join(ROOT, "node_modules/onnxruntime-web/dist")],
  ["/models/", path.join(ROOT, "models")],
  ["/examples/", path.join(ROOT, "examples")],
  ["/", path.join(ROOT, "demo")],
];
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm", ".svg": "image/svg+xml", ".map": "application/json" };

http
  .createServer((req, res) => {
    const url = new URL(req.url, "http://localhost");
    const [prefix, dir] = routes.find(([p]) => url.pathname.startsWith(p));
    let file = path.join(dir, decodeURIComponent(url.pathname.slice(prefix.length)));
    if (!file.startsWith(dir)) return res.writeHead(403).end();
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
    const headers = { "cache-control": "no-cache" };
    if (isolate) {
      headers["cross-origin-opener-policy"] = "same-origin";
      headers["cross-origin-embedder-policy"] = "require-corp";
      headers["cross-origin-resource-policy"] = "same-origin";
    }
    if (!fs.existsSync(file)) return res.writeHead(404, headers).end("not found");
    const size = fs.statSync(file).size;
    res.writeHead(200, { ...headers, "content-type": types[path.extname(file)] || "application/octet-stream", "content-length": size });
    if (req.method === "HEAD") return res.end();
    fs.createReadStream(file).pipe(res);
  })
  .listen(port, () => console.log(`http://localhost:${port}/  (cross-origin isolated: ${isolate})`));
