// Asset loading: integrity, retries, dedupe and the revision-keyed cache.
import test from "node:test";
import assert from "node:assert/strict";
import crypto from "node:crypto";
import { createWebHost, cacheNameFor, pruneOldCaches, AssetIntegrityError } from "../dist/index.js";

const sha = (s) => crypto.createHash("sha256").update(s).digest("hex");
const enc = (s) => new TextEncoder().encode(s);
const manifest = (rev = "abc1234") => ({
  id: `test@${rev}`,
  language: "test",
  repo: "x/y",
  revision: rev,
  modelBaseUrl: `https://models.example/${rev}/test`,
  sampleRate: 24000,
  assets: [{ file: "a.bin", sha256: sha("hello"), bytes: 5, role: "metadata", precision: "none", optional: false }],
  precisions: { fp32: [], "int8-core": [], "int8-wide": [] },
  voices: [{ id: "v", label: "V", file: "voices/v.npy", sha256: sha("voice"), bytes: 5, source: {} }],
});
const ort = { Tensor: class {}, InferenceSession: { create: async (bytes) => ({ bytes }) }, env: { wasm: {} } };

function fakeFetch(responses) {
  const calls = [];
  const fn = async (url) => {
    calls.push(String(url));
    const next = responses.shift();
    if (next instanceof Error) throw next;
    if (typeof next === "function") return next();
    return next;
  };
  fn.calls = calls;
  return fn;
}
const ok = (body) => new Response(enc(body), { status: 200, headers: { "content-length": String(enc(body).length) } });

class FakeCache {
  constructor(store) {
    this.store = store;
  }
  async match(url) {
    const b = this.store.get(url);
    return b ? new Response(b, { headers: { "content-length": String(b.length) } }) : undefined;
  }
  async put(url, res) {
    if (this.failPut) throw new DOMException("quota", "QuotaExceededError");
    this.store.set(url, new Uint8Array(await res.arrayBuffer()));
  }
  async delete(url) {
    return this.store.delete(url);
  }
}
function installCaches() {
  const buckets = new Map();
  globalThis.caches = {
    buckets,
    async open(name) {
      if (!buckets.has(name)) buckets.set(name, new FakeCache(new Map()));
      return buckets.get(name);
    },
    async keys() {
      return [...buckets.keys()];
    },
    async delete(name) {
      return buckets.delete(name);
    },
  };
  return buckets;
}
const flush = () => new Promise((r) => setTimeout(r, 5));

test("verified download resolves from the pinned base URL", async () => {
  const fetch = fakeFetch([ok("hello")]);
  const host = createWebHost({ ort, manifest: manifest(), fetch, cache: false });
  assert.equal(new TextDecoder().decode(await host.readAsset("a.bin", "a")), "hello");
  assert.deepEqual(fetch.calls, ["https://models.example/abc1234/test/a.bin"]);
});

test("a hash mismatch is rejected and never retried into another file", async () => {
  const fetch = fakeFetch([ok("jello"), ok("hello")]);
  const host = createWebHost({ ort, manifest: manifest(), fetch, cache: false, retryDelayMs: 0 });
  await assert.rejects(host.readAsset("a.bin", "a"), (err) => err instanceof AssetIntegrityError && /SHA-256/.test(err.message));
  assert.equal(fetch.calls.length, 1);
});

test("a wrong-size file is rejected", async () => {
  const host = createWebHost({ ort, manifest: manifest(), fetch: fakeFetch([ok("hello!")]), cache: false });
  await assert.rejects(host.readAsset("a.bin", "a"), /expected 5 bytes/);
});

test("a truncated response is retried, then succeeds", async () => {
  const truncated = () => new Response(enc("he"), { status: 200, headers: { "content-length": "5" } });
  const fetch = fakeFetch([truncated, ok("hello")]);
  const host = createWebHost({ ort, manifest: manifest(), fetch, cache: false, retryDelayMs: 0 });
  assert.equal((await host.readAsset("a.bin", "a")).length, 5);
  assert.equal(fetch.calls.length, 2);
});

test("a compressed response (decoded body larger than Content-Length) is accepted", async () => {
  const gzipped = () => new Response(enc("hello"), { status: 200, headers: { "content-length": "3", "content-encoding": "gzip" } });
  const hiddenEncoding = () => new Response(enc("hello"), { status: 200, headers: { "content-length": "3" } });
  for (const res of [gzipped, hiddenEncoding]) {
    const fetch = fakeFetch([res]);
    const host = createWebHost({ ort, manifest: manifest(), fetch, cache: false, retryDelayMs: 0 });
    assert.deepEqual(await host.readAsset("a.bin", "a"), enc("hello"));
    assert.equal(fetch.calls.length, 1);
  }
});

test("429 and network errors are retried a finite number of times", async () => {
  const tooMany = () => new Response(null, { status: 429, headers: { "retry-after": "0" } });
  const fetch = fakeFetch([tooMany, new TypeError("network down"), ok("hello")]);
  const host = createWebHost({ ort, manifest: manifest(), fetch, cache: false, retryDelayMs: 0 });
  assert.equal((await host.readAsset("a.bin", "a")).length, 5);
  assert.equal(fetch.calls.length, 3);

  const exhausted = fakeFetch([tooMany, tooMany, tooMany, ok("hello")]);
  const h2 = createWebHost({ ort, manifest: manifest(), fetch: exhausted, cache: false, retryDelayMs: 0, attempts: 3 });
  await assert.rejects(h2.readAsset("a.bin", "a"), /HTTP 429/);
  assert.equal(exhausted.calls.length, 3);
});

test("404 fails immediately (not transient)", async () => {
  const fetch = fakeFetch([new Response(null, { status: 404 })]);
  const host = createWebHost({ ort, manifest: manifest(), fetch, cache: false, retryDelayMs: 0 });
  await assert.rejects(host.readAsset("a.bin", "a"), /HTTP 404/);
  assert.equal(fetch.calls.length, 1);
});

test("concurrent requests for one asset share a single download", async () => {
  let release;
  const gate = new Promise((r) => (release = r));
  const fetch = fakeFetch([async () => (await gate, ok("hello"))]);
  const host = createWebHost({ ort, manifest: manifest(), fetch, cache: false });
  const a = host.readAsset("a.bin", "a");
  const b = host.readAsset("a.bin", "a");
  release();
  assert.equal((await a).length, 5);
  assert.equal((await b).length, 5);
  assert.equal(fetch.calls.length, 1);
});

test("a prefetched graph is consumed by loadModel without a second download", async () => {
  const fetch = fakeFetch([ok("hello")]);
  const host = createWebHost({ ort, manifest: manifest(), fetch, cache: false });
  host.prefetch("a.bin", "a");
  await flush();
  const session = await host.loadModel({ stem: "flow_lm_main", filename: "a.bin", transform: (b) => b.slice(1) });
  assert.equal(fetch.calls.length, 1);
  assert.equal(new TextDecoder().decode(session.bytes), "ello", "the dynamic-cache transform is applied before session creation");
});

test("stock voices resolve against voiceBaseUrl and are hash-checked", async () => {
  const fetch = fakeFetch([ok("voice")]);
  const host = createWebHost({ ort, manifest: manifest(), fetch, cache: false, voiceBaseUrl: "https://voices.example/" });
  await host.readAsset("voices/v.npy", "voice");
  assert.deepEqual(fetch.calls, ["https://voices.example/voices/v.npy"]);
});

test("cache hit serves offline; cache is keyed by model revision", async () => {
  const buckets = installCaches();
  const m = manifest();
  const host = createWebHost({ ort, manifest: m, fetch: fakeFetch([ok("hello")]) });
  await host.readAsset("a.bin", "a");
  await flush();
  assert.deepEqual([...buckets.keys()], [cacheNameFor(m)]);
  assert.equal(cacheNameFor(m), "pocket-tts:test@abc1234");
  const offline = createWebHost({ ort, manifest: m, fetch: fakeFetch([new TypeError("offline")]) });
  const events = [];
  const again = createWebHost({ ort, manifest: m, fetch: fakeFetch([]), onProgress: (e) => events.push(e) });
  assert.equal((await again.readAsset("a.bin", "a")).length, 5);
  assert.equal(events.at(-1).fromCache, true);
  assert.equal((await offline.readAsset("a.bin", "a")).length, 5);
  delete globalThis.caches;
});

test("a quota failure on cache write does not fail the load", async () => {
  const buckets = installCaches();
  const m = manifest();
  (await caches.open(cacheNameFor(m))).failPut = true;
  const host = createWebHost({ ort, manifest: m, fetch: fakeFetch([ok("hello")]) });
  assert.equal((await host.readAsset("a.bin", "a")).length, 5);
  await flush();
  assert.equal(buckets.get(cacheNameFor(m)).store.size, 0);
  delete globalThis.caches;
});

test("a truncated cache entry is discarded and re-downloaded", async () => {
  installCaches();
  const m = manifest();
  const cache = await caches.open(cacheNameFor(m));
  cache.store.set("https://models.example/abc1234/test/a.bin", enc("hel"));
  const fetch = fakeFetch([ok("hello")]);
  const host = createWebHost({ ort, manifest: m, fetch });
  assert.equal((await host.readAsset("a.bin", "a")).length, 5);
  assert.equal(fetch.calls.length, 1);
  delete globalThis.caches;
});

test("old revisions of the same language are pruned explicitly; other languages kept", async () => {
  const buckets = installCaches();
  await caches.open("pocket-tts:test@old0000");
  await caches.open("pocket-tts:other@old0000");
  await caches.open(cacheNameFor(manifest()));
  await caches.open("unrelated-app-cache");
  const deleted = await pruneOldCaches(manifest());
  assert.deepEqual(deleted, ["pocket-tts:test@old0000"]);
  assert.deepEqual([...buckets.keys()].sort(), ["pocket-tts:other@old0000", "pocket-tts:test@abc1234", "unrelated-app-cache"]);
  delete globalThis.caches;
});
