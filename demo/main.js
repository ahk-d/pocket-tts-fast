// Demo for @flowent/pocket-tts. Uses the built package from /dist (the same
// files npm installs). Query parameters override asset locations:
//   ?models=<url of the folder with bundle.json>  ?voices=<url of the folder with voices/>
//   ?ort=<onnxruntime-web dist url>
// When served by tools/serve.mjs on localhost, local /models and /ort are the defaults.
import { PocketTTS, chunksToWavBlob, startLatency } from "/dist/index.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const local = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
const assets = {
  modelBaseUrl: params.get("models") ?? (local ? `${location.origin}/models/german` : undefined),
  voiceBaseUrl: params.get("voices") ?? (local ? `${location.origin}/models/german` : undefined),
  ortBaseUrl: params.get("ort") ?? (local ? `${location.origin}/ort/` : undefined),
};

const SAMPLES = [
  ["Greeting", "Guten Morgen."],
  ["Question", "Wie spät ist es? Um halb acht fährt der Zug, also beeil dich bitte!"],
  ["Everyday", "Nach dem Unterricht gehe ich in den Laden an der Ecke, weil ich Brot und Obst für das Abendessen brauche."],
  ["Paragraph", "Am Samstag fahren wir früh mit dem Zug in eine kleine Stadt am See. Dort möchten wir zuerst über den Markt gehen, regionale Spezialitäten probieren und danach am Ufer spazieren. Wenn das Wetter hält, setzen wir uns draußen in ein Café und planen in Ruhe unsere nächste gemeinsame Reise."],
];

let tts = null;
let ttsKey = "";
let audioContext = null;
let lastChunks = null;
let busy = false;
const voices = new Map(); // label -> voice ref (per engine instance)
const history = [];

const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "–");
const status = (text) => ($("status").textContent = text);

function settings() {
  return {
    precision: $("precision").value,
    maxThreads: Number($("threads").value) || 4,
    dynamicCache: $("dyncache").checked,
    cache: $("usecache").checked,
  };
}

function setBusy(on) {
  busy = on;
  $("speak").disabled = on;
  $("stop").disabled = !on;
  $("clone").disabled = on || !$("clonefile").files.length;
  for (const id of ["precision", "threads", "dyncache", "usecache"]) $(id).disabled = on;
}

async function engine() {
  const s = settings();
  const key = JSON.stringify(s);
  if (tts && key === ttsKey && tts.ready) return tts;
  tts?.destroy();
  voices.clear();
  tts = new PocketTTS({ ...s, ...assets });
  ttsKey = key;
  const bar = $("progress");
  bar.hidden = false;
  const t0 = performance.now();
  const info = await tts.load((e) => {
    if (e.type === "progress") {
      bar.max = e.total || 1;
      bar.value = e.loaded;
      status(`${e.fromCache ? "Reading" : "Downloading"} ${e.label}: ${(e.loaded / 1e6).toFixed(1)} / ${(e.total / 1e6).toFixed(1)} MB`);
    } else if (e.status === "loading-model") status(`Preparing ${e.stem}…`);
    else if (e.status === "loading-runtime") status("Loading the WebAssembly runtime…");
  });
  bar.hidden = true;
  const threads = info.crossOriginIsolated ? `${info.actualThreads} of ${info.requestedThreads} requested` : "1 (page is not cross-origin isolated)";
  status(`Ready in ${fmt((performance.now() - t0) / 1000, 1)} s · ${info.modelId} · ${info.precision} · threads: ${threads}. Voice changes reuse this model.`);
  return tts;
}

async function voiceRef(t) {
  const choice = $("voice").value;
  if (voices.has(choice)) return voices.get(choice);
  if (!choice.startsWith("stock:")) throw new Error("Prepare the cloned voice again after changing model settings.");
  status("Preparing voice…");
  const ref = await t.useVoice(choice.slice(6));
  voices.set(choice, ref);
  return ref;
}

async function speak() {
  const text = $("text").value.trim();
  if (!text) return status("Type some text first.");
  setBusy(true);
  try {
    const t = await engine();
    const voice = await voiceRef(t);
    audioContext ??= new AudioContext();
    await audioContext.resume();
    const requestStartedMs = performance.now();
    let latency = null;
    status("Generating…");
    const result = await t.speak(text, {
      voice,
      audioContext,
      mode: $("mode").value,
      onPlaybackStart: (player) => {
        latency = startLatency({
          requestStartedMs,
          generationStartedMs: requestStartedMs,
          snapshotPerformanceMs: performance.now(),
          snapshotAudioSeconds: audioContext.currentTime,
          scheduledStartSeconds: player.startedAt,
        });
        status("Playing…");
      },
    });
    const wall = (performance.now() - requestStartedMs) / 1000;
    lastChunks = result.chunks;
    const audioSeconds = result.chunks.reduce((n, c) => n + c.length, 0) / t.sampleRate;
    const row = {
      n: history.length + 1,
      text,
      voice: $("voice").selectedOptions[0].textContent,
      precision: t.info.precision,
      dynamicCache: t.info.dynamicCache,
      threads: t.info.actualThreads,
      crossOriginIsolated: t.info.crossOriginIsolated,
      mode: $("mode").value,
      startSeconds: latency?.requestToPlaybackSeconds ?? null,
      generationSeconds: wall,
      audioSeconds,
      rtf: wall / audioSeconds,
      gaps: result.player.gaps,
      gapSeconds: result.player.gapSeconds,
      stopped: result.metrics.stopped,
      pcmSha256: await pcmHash(result.chunks),
      userAgent: navigator.userAgent,
      at: new Date().toISOString(),
    };
    history.push(row);
    render(row);
    status(row.stopped ? "Stopped." : "Done. Playing out…");
    $("replay").disabled = $("download").disabled = false;
    await result.played;
    if (!busy) return;
    status(row.stopped ? "Stopped." : "Done.");
  } catch (err) {
    console.error(err);
    status(`Error: ${err.message}`);
  } finally {
    setBusy(false);
  }
}

async function pcmHash(chunks) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const all = new Float32Array(total);
  let o = 0;
  for (const c of chunks) all.set(c, (o += c.length) - c.length);
  const digest = await crypto.subtle.digest("SHA-256", all.buffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function render(r) {
  const items = [
    ["Starts speaking", r.startSeconds == null ? "–" : `${fmt(r.startSeconds)} s`],
    ["Generation", `${fmt(r.generationSeconds)} s`],
    ["Audio duration", `${fmt(r.audioSeconds)} s`],
    ["RTF", fmt(r.rtf), r.rtf < 1 ? "good" : "bad"],
    ["Playback gaps", `${r.gaps} (${fmt(r.gapSeconds)} s)`, r.gaps ? "bad" : "good"],
    ["Threads", `${r.threads}${r.crossOriginIsolated ? "" : " (not isolated)"}`],
    ["Precision", r.precision],
    ["KV cache", r.dynamicCache ? "dynamic" : "fixed 1000"],
  ];
  $("metrics").replaceChildren(
    ...items.map(([k, v, cls]) => {
      const div = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = k;
      dd.textContent = v;
      if (cls) dd.className = cls;
      div.append(dt, dd);
      return div;
    }),
  );
  const tr = document.createElement("tr");
  for (const v of [r.n, r.voice, r.precision, r.dynamicCache ? "dynamic" : "fixed", r.threads, r.mode, r.startSeconds == null ? "–" : fmt(r.startSeconds), fmt(r.generationSeconds), fmt(r.audioSeconds), fmt(r.rtf), r.gaps, r.pcmSha256.slice(0, 12)]) {
    const td = document.createElement("td");
    td.textContent = v;
    tr.append(td);
  }
  $("history").tBodies[0].append(tr);
}

async function stop() {
  status("Stopping…");
  await tts?.stop();
}

async function replay() {
  if (!lastChunks) return;
  audioContext ??= new AudioContext();
  await audioContext.resume();
  const total = lastChunks.reduce((n, c) => n + c.length, 0);
  const buffer = audioContext.createBuffer(1, total, tts.sampleRate);
  let o = 0;
  for (const c of lastChunks) buffer.copyToChannel(c, 0, (o += c.length) - c.length);
  const src = audioContext.createBufferSource();
  src.buffer = buffer;
  src.connect(audioContext.destination);
  src.start();
}

function download() {
  if (!lastChunks) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(chunksToWavBlob(lastChunks, tts.sampleRate));
  a.download = `pocket-tts-${Date.now()}.wav`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function clone() {
  const file = $("clonefile").files[0];
  if (!file) return;
  setBusy(true);
  try {
    const t = await engine();
    status("Decoding the clip locally…");
    const { decodeAudioFile } = await import("/dist/index.js");
    const { audio, sampleRate } = await decodeAudioFile(file);
    status("Encoding the voice on this device (first time loads the encoder)…");
    const t0 = performance.now();
    const ref = await t.cloneVoice(audio, { inputSampleRate: sampleRate });
    const label = `Cloned: ${file.name}`;
    const key = `clone:${voices.size}`;
    voices.set(key, ref);
    const opt = new Option(label, key);
    $("voice").append(opt);
    $("voice").value = key;
    status(`Cloned voice ready in ${fmt((performance.now() - t0) / 1000, 1)} s.`);
  } catch (err) {
    status(`Error: ${err.message}`);
  } finally {
    setBusy(false);
  }
}

function exportResults() {
  const blob = new Blob([JSON.stringify({ assets, history }, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "pocket-tts-results.json";
  a.click();
}

for (const v of new PocketTTS().voices) $("voice").append(new Option(v.label, `stock:${v.id}`));
for (const [label, text] of SAMPLES) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = label;
  b.onclick = () => ($("text").value = text);
  $("samples").append(b);
}
$("speak").onclick = speak;
$("stop").onclick = stop;
$("replay").onclick = replay;
$("download").onclick = download;
$("clone").onclick = clone;
$("export").onclick = exportResults;
$("clonefile").onchange = () => ($("clone").disabled = busy || !$("clonefile").files.length);
if (!local && !params.get("voices")) {
  status("Note: stock voice files are not hosted publicly yet; pass ?voices=<url> or clone a voice from a file.");
}
window.__demo = { get tts() { return tts; }, history };
