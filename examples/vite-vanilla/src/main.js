import { PocketTTS } from "@flowent/pocket-tts";

const tts = new PocketTTS({
  modelBaseUrl: import.meta.env.VITE_MODEL_BASE_URL,
  voiceBaseUrl: import.meta.env.VITE_VOICE_BASE_URL,
  ortBaseUrl: import.meta.env.VITE_ORT_BASE_URL,
});
const status = (s) => (document.getElementById("status").textContent = s);
let voice;
document.getElementById("speak").onclick = async () => {
  if (!tts.ready) await tts.load((e) => e.type === "progress" && status(`${e.label} ${Math.round((100 * e.loaded) / e.total)}%`));
  voice ??= await tts.useVoice("alba");
  const { played } = await tts.speak(document.getElementById("text").value, { voice });
  await played;
  status("done");
};
