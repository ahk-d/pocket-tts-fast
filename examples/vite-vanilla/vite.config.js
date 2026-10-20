import { defineConfig } from "vite";

export default defineConfig({
  // Keep the package out of dependency pre-bundling so its
  // `new Worker(new URL("./worker.js", import.meta.url))` resolves in dev.
  optimizeDeps: { exclude: ["pocket-tts-fast"] },
  // Cross-origin isolation enables WASM threads (otherwise 1 thread).
  server: { headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" } },
  preview: { headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" } },
});
