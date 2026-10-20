import test from "node:test";
import assert from "node:assert/strict";

test("importing the package constructs no Worker and downloads nothing (SSR-safe)", async () => {
  let workers = 0;
  const realFetch = globalThis.fetch;
  const realWorker = globalThis.Worker;
  globalThis.Worker = class {
    constructor() {
      workers++;
    }
  };
  globalThis.fetch = async () => {
    throw new Error("import attempted a download");
  };
  try {
    const mod = await import(`../dist/index.js?ssr=${Date.now()}`);
    const tts = new mod.PocketTTS();
    assert.equal(workers, 0, "constructing an instance does not start the worker either");
    assert.equal(tts.ready, false);
    assert.equal(typeof mod.PocketTTS, "function");
  } finally {
    globalThis.fetch = realFetch;
    globalThis.Worker = realWorker;
  }
});

test("the published worker bundle is self-contained (no bare imports)", async () => {
  const { readFile } = await import("node:fs/promises");
  for (const file of ["../dist/index.js", "../dist/worker.js"]) {
    const src = await readFile(new URL(file, import.meta.url), "utf8");
    assert.doesNotMatch(src, /^\s*import\s[^;]*from\s*["'](?![./])/m, `${file} has a bare import`);
    assert.doesNotMatch(src, /\/Users\//, `${file} contains a machine-local path`);
    assert.doesNotMatch(src, /onnxruntime-web@latest|@latest\//, `${file} references a moving version`);
  }
});
