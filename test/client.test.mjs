// Worker lifecycle contract (ported and extended from
// browser-quant-v1/web/runtime/test/lifecycle.test.js).
import test from "node:test";
import assert from "node:assert/strict";
import { PocketTTS, BUILD_ID } from "../dist/index.js";
import pkg from "../package.json" with { type: "json" };

const workers = [];
class MockWorker {
  constructor(url, opts) {
    this.url = String(url);
    this.opts = opts;
    this.messages = [];
    this.terminated = false;
    workers.push(this);
  }
  postMessage(message) {
    this.messages.push(message);
  }
  terminate() {
    this.terminated = true;
  }
  emit(data) {
    this.onmessage({ data });
  }
  crash(message = "worker crashed") {
    this.onerror({ message });
  }
  last(type) {
    return this.messages.findLast((m) => m.type === type);
  }
}
const info = { build: BUILD_ID, language: "german", modelId: "german@58a6d00", precision: "int8-wide", sampleRate: 24000, loadedFiles: [], dynamicCache: true, requestedThreads: 4, actualThreads: 4, crossOriginIsolated: true, executionProvider: "wasm", ortVersion: "1.20.0" };

function setup() {
  workers.length = 0;
  globalThis.Worker = MockWorker;
}
async function loaded(options) {
  setup();
  const tts = new PocketTTS(options);
  const load = tts.load();
  const w = workers[0];
  w.emit({ id: w.last("init").id, type: "result", result: info });
  await load;
  return { tts, w };
}

test("operations fail fast before a successful load and create no worker", async () => {
  setup();
  const tts = new PocketTTS();
  await assert.rejects(tts.generate("Hallo.", { voice: "stock:alba" }), /load/i);
  await assert.rejects(tts.cloneVoice(new Float32Array([0.1])), /load/i);
  await assert.rejects(tts.useVoice("alba"), /load/i);
  await assert.rejects(tts.speak("Hallo.", { voice: "stock:alba" }), /load/i);
  assert.equal(workers.length, 0);
});

test("constructor validates language and precision and sends the pinned config", async () => {
  setup();
  assert.throws(() => new PocketTTS({ language: "klingon" }), /Unsupported language/);
  assert.throws(() => new PocketTTS({ precision: "int4" }), /Unknown precision/);
  const { tts, w } = await loaded({ precision: "fp32" });
  const cfg = w.messages[0].payload;
  assert.equal(cfg.precision, "fp32");
  assert.equal(cfg.dynamicCache, true);
  assert.equal(cfg.maxThreads, 4);
  assert.match(cfg.ortBaseUrl, /onnxruntime-web@1\.20\.0\/dist\/$/);
  assert.equal(w.opts.type, "module");
  assert.match(w.url, /worker\.js$/);
  assert.equal(tts.ready, true);
  tts.destroy();
});

test("concurrent load() calls share one worker and one init", async () => {
  setup();
  const tts = new PocketTTS();
  const events = [];
  const a = tts.load((e) => events.push(e));
  const b = tts.load();
  assert.equal(workers.length, 1);
  const w = workers[0];
  assert.equal(w.messages.filter((m) => m.type === "init").length, 1);
  w.emit({ type: "progress", label: "flow_lm_main", loaded: 1, total: 2, fromCache: false });
  w.emit({ id: w.last("init").id, type: "result", result: info });
  assert.deepEqual(await a, info);
  assert.deepEqual(await b, info);
  assert.equal(events.length, 1);
  // Later loads resolve immediately without another init.
  await tts.load();
  assert.equal(w.messages.filter((m) => m.type === "init").length, 1);
  tts.destroy();
});

test("a failed init resets so the next load() starts a fresh worker", async () => {
  setup();
  const tts = new PocketTTS();
  const first = tts.load();
  workers[0].emit({ id: workers[0].last("init").id, type: "error", error: "SHA-256 mismatch" });
  await assert.rejects(first, /SHA-256/);
  assert.equal(workers[0].terminated, true);
  const second = tts.load();
  assert.equal(workers.length, 2);
  workers[1].emit({ id: workers[1].last("init").id, type: "result", result: info });
  await second;
  tts.destroy();
});

test("switching stock voices reuses the loaded model (no second init)", async () => {
  const { tts, w } = await loaded();
  for (const id of ["alba", "alba"]) {
    const p = tts.useVoice(id);
    const req = w.last("useVoice");
    assert.equal(req.payload.id, id);
    w.emit({ id: req.id, type: "result", result: { ref: `stock:${id}` } });
    assert.equal(await p, "stock:alba");
  }
  assert.equal(w.messages.filter((m) => m.type === "init").length, 1);
  assert.equal(workers.length, 1);
  tts.destroy();
});

test("worker errors invalidate the runtime and allow a fresh load", async () => {
  const { tts, w } = await loaded();
  const voice = tts.useVoice("alba");
  w.crash("wasm fault");
  await assert.rejects(voice, /wasm fault/);
  assert.equal(w.terminated, true);
  assert.equal(tts.ready, false);
  await assert.rejects(tts.generate("Hallo.", { voice: "stock:alba" }), /load/i);
  const next = tts.load();
  const replacement = workers[1];
  assert.notEqual(replacement, w);
  replacement.emit({ id: replacement.last("init").id, type: "result", result: info });
  await next;
  tts.destroy();
});

test("overlapping generation is rejected and destroy rejects pending requests", async () => {
  const { tts, w } = await loaded();
  const first = tts.generate("Hallo.", { voice: "stock:alba" });
  assert.equal(w.messages.at(-1).type, "generate");
  await assert.rejects(tts.generate("Noch einmal.", { voice: "stock:alba" }), /already in progress/i);
  tts.destroy();
  await assert.rejects(first, /destroy/i);
  assert.equal(w.terminated, true);
  assert.equal(tts.ready, false);
  tts.destroy(); // idempotent
});

test("chunks stream to onChunk and delivery size reaches the worker", async () => {
  const { tts, w } = await loaded();
  const got = [];
  const live = tts.generate("Hallo.", { voice: "stock:alba", deliveryFrames: 3, onChunk: (a, m) => got.push([a.length, m.isFirst]) });
  let req = w.last("generate");
  assert.equal(req.payload.deliveryFrames, 3);
  w.emit({ type: "chunk", audio: new Float32Array(5760), meta: { isFirst: true } });
  w.emit({ id: req.id, type: "result", result: { metrics: { stopped: false, audioDuration: 0.24 } } });
  await live;
  assert.deepEqual(got, [[5760, true]]);
  const benchmark = tts.generate("Hallo.", { voice: "stock:alba" });
  req = w.last("generate");
  assert.equal(req.payload.deliveryFrames, 12);
  w.emit({ id: req.id, type: "result", result: { metrics: { stopped: false } } });
  await benchmark;
  await assert.rejects(tts.generate("Hallo.", { voice: "stock:alba", deliveryFrames: 0 }), /delivery/i);
  tts.destroy();
});

test("stop during priming cancels playback and the generation", async () => {
  const { tts, w } = await loaded();
  const sources = [];
  const ctx = {
    currentTime: 0,
    destination: {},
    createBuffer: (_c, n, r) => ({ duration: n / r, copyToChannel() {} }),
    createBufferSource: () => {
      const s = { connect() {}, start() {}, stop() {} };
      sources.push(s);
      return s;
    },
  };
  const speaking = tts.speak("Ein längerer Satz.", { voice: "stock:alba", audioContext: ctx }); // queued mode, 1 s prime
  await new Promise((r) => setImmediate(r));
  const req = w.last("generate");
  assert.equal(req.payload.deliveryFrames, 3);
  w.emit({ type: "chunk", audio: new Float32Array(5760), meta: {} }); // 240 ms: still priming
  assert.equal(sources.length, 0);
  const stopping = tts.stop();
  const stopReq = w.last("stop");
  w.emit({ id: stopReq.id, type: "result", result: { ok: true } });
  await stopping;
  w.emit({ type: "chunk", audio: new Float32Array(24000), meta: {} }); // arrives after stop
  w.emit({ id: req.id, type: "result", result: { metrics: { stopped: true } } });
  const { player, chunks } = await speaking;
  assert.equal(player.cancelled, true);
  assert.equal(sources.length, 0, "nothing was scheduled");
  assert.equal(chunks.length, 2, "raw output is kept separately from playback");
  tts.destroy();
});

test("a worker from another build is refused (stale cached worker.js)", async () => {
  setup();
  const tts = new PocketTTS();
  const load = tts.load();
  const w = workers[0];
  w.emit({ id: w.last("init").id, type: "result", result: { ...info, build: "0.0.0+stale" } });
  await assert.rejects(load, /does not match client build/);
  assert.equal(tts.ready, false);
  assert.equal(w.terminated, true);
  assert.ok(BUILD_ID.startsWith(`${pkg.version}+`) && /\+[0-9a-f]{12}$/.test(BUILD_ID), BUILD_ID);
});
