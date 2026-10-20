// A small hook over the framework-independent package. Copy and adapt it;
// the package itself has no React dependency.
import { useCallback, useEffect, useRef, useState } from "react";
import { PocketTTS } from "@flowent/pocket-tts";

export function usePocketTTS(options) {
  const ref = useRef(null);
  const [state, setState] = useState({ status: "idle", progress: null, error: null, info: null });

  // One engine per component lifetime; destroyed on unmount. Generation is client-only.
  useEffect(() => {
    const tts = new PocketTTS(options);
    ref.current = tts;
    return () => {
      tts.destroy();
      ref.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(options)]);

  const ensureLoaded = useCallback(async () => {
    const tts = ref.current;
    if (tts.ready) return tts;
    setState((s) => ({ ...s, status: "loading", error: null }));
    const info = await tts.load((e) => e.type === "progress" && setState((s) => ({ ...s, progress: e })));
    setState((s) => ({ ...s, status: "ready", progress: null, info }));
    return tts;
  }, []);

  const speak = useCallback(
    async (text, { voice = "alba", mode = "queued" } = {}) => {
      try {
        const tts = await ensureLoaded();
        const ref = await tts.useVoice(voice);
        setState((s) => ({ ...s, status: "speaking" }));
        const result = await tts.speak(text, { voice: ref, mode });
        await result.played;
        setState((s) => ({ ...s, status: "ready" }));
        return result;
      } catch (error) {
        setState((s) => ({ ...s, status: "error", error }));
        throw error;
      }
    },
    [ensureLoaded],
  );

  const stop = useCallback(() => ref.current?.stop(), []);
  return { ...state, speak, stop };
}
