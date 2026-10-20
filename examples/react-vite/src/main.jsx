import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { usePocketTTS } from "./usePocketTTS.js";

function App() {
  const [text, setText] = useState("Guten Morgen! Wie geht es dir heute?");
  const { status, progress, error, info, speak, stop } = usePocketTTS({
    // Optional overrides for self-hosted assets (defaults: pinned Hugging Face + jsDelivr).
    modelBaseUrl: import.meta.env.VITE_MODEL_BASE_URL,
    voiceBaseUrl: import.meta.env.VITE_VOICE_BASE_URL ?? "/",
    ortBaseUrl: import.meta.env.VITE_ORT_BASE_URL,
  });
  return (
    <main style={{ maxWidth: 640, margin: "2rem auto", padding: "0 16px", fontFamily: "system-ui" }}>
      <h1>Pocket TTS + React</h1>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} style={{ width: "100%" }} />
      <p>
        <button onClick={() => speak(text).catch(() => {})} disabled={status === "loading" || status === "speaking"}>Speak</button>{" "}
        <button onClick={stop} disabled={status !== "speaking"}>Stop</button>
      </p>
      <p role="status">
        {status}
        {progress && ` · ${progress.label} ${Math.round((100 * progress.loaded) / progress.total)}%`}
        {info && ` · ${info.precision}, ${info.actualThreads} thread(s)`}
        {error && ` · ${error.message}`}
      </p>
    </main>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
