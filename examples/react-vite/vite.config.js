import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };

export default defineConfig({
  plugins: [react()],
  optimizeDeps: { exclude: ["@flowent/pocket-tts"] },
  server: { headers: isolation },
  preview: { headers: isolation },
});
