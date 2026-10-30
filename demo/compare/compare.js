// Side-by-side comparison: the same phrases generated with the stock fixed
// 1000-slot KV cache and with the dynamic cache, in this browser tab. Modes run
// one after the other (never concurrently) with one engine loaded at a time.
// Query parameters override asset locations, as in the main demo:
//   ?models=<folder with bundle.json>  ?voices=<folder containing voices/>  ?ort=<onnxruntime-web dist>
// and the benchmark itself:
//   ?runs=<1-5>  timed runs per phrase and mode (median reported)   ?set=reel  the phrases used in the video
import { PocketTTS } from "/dist/index.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const local = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);

const PHRASES = {
  german: [
    "Danke schön.",
    "Wo ist der Bahnhof, bitte?",
    "Ich hätte gern einen Kaffee mit Milch und ein Stück Apfelkuchen.",
    "Heute regnet es den ganzen Tag, deshalb bleiben wir zu Hause und kochen zusammen eine Suppe.",
    "Am Wochenende fahren wir mit dem Zug an einen kleinen See. Dort wollen wir wandern, im Café sitzen und abends am Ufer den Sonnenuntergang anschauen.",
  ],
  english: [
    "Thank you very much.",
    "Excuse me, where is the train station?",
    "I'd like a coffee with milk and a slice of apple cake, please.",
    "It's raining all day today, so we're staying home and cooking a big pot of soup together.",
    "This weekend we're taking the train to a small lake. We want to go hiking, sit in a little café, and watch the sunset from the shore in the evening.",
  ],
};
if (params.get("set") === "reel") {
  PHRASES.english = [
    "Okay, that was fast.",
    "I had a very productive morning, if thinking about work counts as work.",
    "I opened my laptop to fix one small problem. Two hours later, I'd rearranged my entire desk, made another coffee, and completely forgotten what I was trying to fix.",
  ];
  PHRASES.german = [
    "Moment, ich kann auch Deutsch.",
    "Ich wollte nur kurz etwas ausprobieren, und jetzt sitze ich seit drei Stunden am Computer.",
  ];
}
const RUNS = Math.min(5, Math.max(1, Math.floor(Number(params.get("runs"))) || 1));
const WARMUP = { german: "Hallo.", english: "Hello." };
const MODES = [
  { key: "fixed", dynamicCache: false, lane: $("lane-fixed") },
  { key: "dynamic", dynamicCache: true, lane: $("lane-dynamic") },
];

let language = "german";
let running = false;
let audioContext = null;

// On a static host the site can live under a sub-path (GitHub Pages serves
// /<repo>/), so voices/ and ort/ resolve next to this script, not at the origin.
const site = new URL(".", import.meta.url).href.replace(/\/$/, "");
function assets(lang) {
  return {
    modelBaseUrl: params.get("models") ?? (local ? `${location.origin}/models/${lang}` : undefined),
    voiceBaseUrl: params.get("voices") ?? (local ? `${location.origin}/models/${lang}` : site),
    ortBaseUrl: params.get("ort") ?? (local ? `${location.origin}/ort/` : `${site}/ort/`),
  };
}

const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "–");
const status = (t) => ($("status").textContent = t);
const median = (xs) => {
  const s = xs.filter(Number.isFinite).sort((a, b) => a - b);
  if (!s.length) return NaN;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

function showPhrases() {
  $("phrases").replaceChildren(...PHRASES[language].map((p) => Object.assign(document.createElement("li"), { textContent: p })));
}

for (const b of document.querySelectorAll(".seg button")) {
  b.onclick = () => {
    if (running) return;
    language = b.dataset.lang;
    for (const o of document.querySelectorAll(".seg button")) o.setAttribute("aria-checked", String(o === b));
    showPhrases();
  };
}

// --- live lane view -------------------------------------------------------
function lane(mode) {
  const el = mode.lane;
  const clock = el.querySelector("[data-clock]");
  const canvas = el.querySelector("[data-wave]");
  const color = getComputedStyle(el.querySelector(".clock")).color;
  let peaks = [];
  let timer = 0;
  const draw = () => {
    const w = (canvas.width = canvas.clientWidth * devicePixelRatio);
    const h = (canvas.height = 64 * devicePixelRatio);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = color;
    const bar = Math.max(2, w / 160);
    const n = Math.floor(w / bar);
    const step = Math.max(1, peaks.length / n);
    for (let i = 0, x = 0; i < peaks.length && x < w; i += step, x += bar) {
      const p = Math.min(1, peaks[Math.floor(i)] * 3);
      const bh = Math.max(1, p * h);
      ctx.fillRect(x, (h - bh) / 2, bar * 0.7, bh);
    }
  };
  return {
    start() {
      peaks = [];
      draw();
      const t0 = performance.now();
      cancelAnimationFrame(timer);
      const tick = () => {
        clock.textContent = `${fmt((performance.now() - t0) / 1000)} s`;
        timer = requestAnimationFrame(tick);
      };
      tick();
    },
    chunk(pcm) {
      for (let i = 0; i < pcm.length; i += 480) {
        let m = 0;
        for (let j = i; j < Math.min(i + 480, pcm.length); j++) m = Math.max(m, Math.abs(pcm[j]));
        peaks.push(m);
      }
      draw();
    },
    stop(seconds) {
      cancelAnimationFrame(timer);
      clock.textContent = `${fmt(seconds)} s`;
    },
  };
}

// --- measurement ----------------------------------------------------------
async function pcmHash(chunks) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const all = new Float32Array(total);
  let o = 0;
  for (const c of chunks) all.set(c, (o += c.length) - c.length);
  const digest = await crypto.subtle.digest("SHA-256", all.buffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function runMode(mode, phrases) {
  const view = lane(mode);
  const tts = new PocketTTS({ language, dynamicCache: mode.dynamicCache, maxThreads: 4, ...assets(language) });
  const bar = $("progress");
  bar.hidden = false;
  const info = await tts.load((e) => {
    if (e.type === "progress") {
      bar.max = e.total || 1;
      bar.value = e.loaded;
      status(`${e.fromCache ? "Reading" : "Downloading"} ${e.label}: ${(e.loaded / 1e6).toFixed(0)} / ${(e.total / 1e6).toFixed(0)} MB`);
    } else if (e.status === "loading-runtime") status("Loading the WebAssembly runtime…");
    else if (e.status === "loading-model") status(`Preparing ${e.stem}…`);
  });
  bar.hidden = true;
  const voice = await tts.useVoice("alba");
  status(`${mode.key === "fixed" ? "Original" : "Fast"}: warming up…`);
  await tts.generate(WARMUP[language], { voice });
  const rows = [];
  for (const [i, text] of phrases.entries()) {
    const runs = [];
    for (let r = 0; r < RUNS; r++) {
      status(`${mode.key === "fixed" ? "Original" : "Fast"}: phrase ${i + 1} of ${phrases.length}${RUNS > 1 ? `, run ${r + 1} of ${RUNS}` : ""}`);
      const chunks = [];
      let first = null;
      view.start();
      const t0 = performance.now();
      await tts.generate(text, {
        voice,
        onChunk: (pcm, meta) => {
          if (first == null && !meta.isSilence) first = (performance.now() - t0) / 1000;
          chunks.push(pcm.slice());
          view.chunk(pcm);
        },
      });
      const gen = (performance.now() - t0) / 1000;
      view.stop(gen);
      runs.push({ gen, first, chunks, hash: await pcmHash(chunks) });
    }
    // Every run must produce the same samples; a mismatch shows up as a differing hash.
    const hash = runs.every((r) => r.hash === runs[0].hash) ? runs[0].hash : "varies";
    const gen = median(runs.map((r) => r.gen));
    const chunks = runs[0].chunks;
    const audio = chunks.reduce((n, c) => n + c.length, 0) / tts.sampleRate;
    rows.push({ text, gen, first: median(runs.map((r) => r.first)), audio, rtf: gen / audio, chunks, hash, sampleRate: tts.sampleRate });
  }
  tts.destroy();
  return { info, rows };
}

function play(row) {
  audioContext ??= new AudioContext();
  audioContext.resume();
  const total = row.chunks.reduce((n, c) => n + c.length, 0);
  const buf = audioContext.createBuffer(1, total, row.sampleRate);
  let o = 0;
  for (const c of row.chunks) buf.copyToChannel(c, 0, (o += c.length) - c.length);
  const src = audioContext.createBufferSource();
  src.buffer = buf;
  src.connect(audioContext.destination);
  src.start();
}

function render(fixed, dynamic) {
  const body = $("results").tBodies[0];
  body.replaceChildren();
  fixed.rows.forEach((f, i) => {
    const d = dynamic.rows[i];
    const tr = document.createElement("tr");
    const same = f.hash === d.hash && f.hash !== "varies";
    const cells = [
      f.text,
      `${fmt(f.gen)} s`,
      `${fmt(d.gen)} s`,
      `${fmt(f.gen / d.gen)}×`,
      `${fmt(f.first)} → ${fmt(d.first)} s`,
      `${fmt(f.rtf)} → ${fmt(d.rtf)}`,
      same ? "yes" : "no",
    ];
    cells.forEach((v, j) => {
      const td = document.createElement("td");
      td.textContent = v;
      if (j === 3 || (j === 6 && same)) td.className = "good";
      tr.append(td);
    });
    const td = document.createElement("td");
    for (const [label, row] of [["Original", f], ["Fast", d]]) {
      const b = Object.assign(document.createElement("button"), { type: "button", textContent: label });
      b.onclick = () => play(row);
      td.append(b, " ");
    }
    tr.append(td);
    body.append(tr);
  });
  const speed = median(fixed.rows.map((f, i) => f.gen / dynamic.rows[i].gen));
  const same = fixed.rows.filter((f, i) => f.hash === dynamic.rows[i].hash && f.hash !== "varies").length;
  $("sum-speed").textContent = `${fmt(speed, 1)}×`;
  $("sum-first").textContent = `${fmt(median(fixed.rows.map((r) => r.first)))} → ${fmt(median(dynamic.rows.map((r) => r.first)))} s`;
  $("sum-same").textContent = `${same} / ${fixed.rows.length}`;
  $("summary").hidden = false;
  const i = dynamic.info;
  $("device").textContent = `${i.crossOriginIsolated ? `${i.actualThreads} WASM threads` : "1 WASM thread (page not cross-origin isolated)"} · ${i.precision} · ${RUNS > 1 ? `median of ${RUNS} runs` : "1 run"} per phrase · ${navigator.hardwareConcurrency ?? "?"} logical cores · ${navigator.userAgent}`;
}

$("run").onclick = async () => {
  if (running) return;
  running = true;
  $("run").disabled = true;
  for (const b of document.querySelectorAll(".seg button")) b.disabled = true;
  $("summary").hidden = true;
  $("results").tBodies[0].replaceChildren();
  try {
    const phrases = PHRASES[language];
    const fixed = await runMode(MODES[0], phrases);
    const dynamic = await runMode(MODES[1], phrases);
    render(fixed, dynamic);
    status("Done. Play any row to compare by ear.");
  } catch (err) {
    console.error(err);
    status(`Error: ${err.message}`);
  } finally {
    running = false;
    $("run").disabled = false;
    for (const b of document.querySelectorAll(".seg button")) b.disabled = false;
  }
};

showPhrases();
window.__compare = { PHRASES };
