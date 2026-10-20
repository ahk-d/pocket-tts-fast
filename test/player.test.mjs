// Ported from browser-quant-v1/test_stream_player.mjs (MeasuredStreamPlayer -> StreamPlayer).
import test from "node:test";
import assert from "node:assert/strict";
import { StreamPlayer, scheduledReplay, adaptiveStartSeconds, estimateSpeechSeconds } from "../dist/index.js";

function context() {
  return {
    currentTime: 10,
    destination: {},
    sources: [],
    createBuffer(_c, n, r) {
      return { duration: n / r, copyToChannel() {} };
    },
    createBufferSource() {
      const s = { connect() {}, start(t) { this.startAt = t; }, stop() { this.onended?.(); } };
      this.sources.push(s);
      return s;
    },
  };
}
const pcm = (n) => new Float32Array(n).fill(0.1);

test("immediate start and actual late scheduling gaps, ignoring natural silence", async () => {
  const ctx = context();
  const p = new StreamPlayer(ctx, { sampleRate: 100, primeSeconds: 0 });
  p.play(pcm(24));
  assert.equal(p.events[0].startAt, 10.05);
  ctx.currentTime = 10.1;
  p.play(pcm(100), { isSilence: true });
  assert.equal(p.events[1].startAt, 10.290000000000001);
  ctx.currentTime = 12;
  p.play(pcm(50));
  assert.equal(p.gaps, 1);
  assert.ok(Math.abs(p.gapSeconds - 0.71) < 1e-8);
  const done = p.finish();
  let finished = false;
  done.then(() => (finished = true));
  await Promise.resolve();
  assert.equal(finished, false);
  for (const s of ctx.sources) s.onended();
  await done;
});

test("one-second prime holds the first 240 ms block; finish releases a short utterance", async () => {
  const ctx = context();
  const p = new StreamPlayer(ctx, { sampleRate: 24000, primeSeconds: 1 });
  p.play(new Float32Array(5760));
  assert.equal(ctx.sources.length, 0);
  assert.equal(p.startedAt, null);
  const done = p.finish();
  assert.equal(ctx.sources.length, 1);
  for (const s of ctx.sources) s.onended();
  await done;
});

test("two-second prime buffers and finish releases short utterances", async () => {
  const ctx = context();
  const p = new StreamPlayer(ctx, { sampleRate: 100, primeSeconds: 2 });
  p.play(pcm(50));
  ctx.currentTime = 11;
  p.play(pcm(50));
  assert.equal(ctx.sources.length, 0);
  const done = p.finish();
  assert.equal(ctx.sources.length, 2);
  assert.equal(p.events[0].startAt, 11.05);
  for (const s of ctx.sources) s.onended();
  await done;
});

test("buffer threshold releases at two seconds and stop releases completion", async () => {
  const ctx = context();
  const p = new StreamPlayer(ctx, { sampleRate: 100, primeSeconds: 2 });
  p.play(pcm(100));
  p.play(pcm(100));
  assert.equal(ctx.sources.length, 2);
  p.stop();
  await p.finish();
  assert.equal(p.cancelled, true);
});

test("stop while priming discards the queue and never starts playback", async () => {
  const ctx = context();
  const p = new StreamPlayer(ctx, { sampleRate: 100, primeSeconds: 2 });
  p.play(pcm(50));
  p.stop();
  p.play(pcm(500));
  await p.finish();
  assert.equal(ctx.sources.length, 0);
});

test("waitForComplete holds stalled chunks until finish, then schedules them contiguously", async () => {
  const ctx = context();
  const p = new StreamPlayer(ctx, { sampleRate: 1, primeSeconds: 2, waitForComplete: true });
  p.play(new Float32Array([0.1, 0.2]));
  ctx.currentTime = 15;
  p.play(new Float32Array([0.3, 0.4]));
  ctx.currentTime = 27;
  p.play(new Float32Array([0.5]));
  assert.equal(ctx.sources.length, 0);
  const done = p.finish();
  assert.equal(ctx.sources.length, 3);
  assert.deepEqual(p.events.map((e) => e.startAt), [27.05, 29.05, 31.05]);
  const replay = scheduledReplay([new Float32Array([0.1, 0.2]), new Float32Array([0.3, 0.4]), new Float32Array([0.5])], p.events, 1);
  assert.deepEqual([...replay].map((v) => Math.round(v * 100) / 100), [0.1, 0.2, 0.3, 0.4, 0.5]);
  for (const s of ctx.sources) s.onended();
  await done;
});

test("scheduled replay retains late gaps but omits initial startup wait", () => {
  const chunks = [new Float32Array([0.1, 0.2]), new Float32Array([0.3])];
  const result = scheduledReplay(chunks, [{ startAt: 10, frames: 2 }, { startAt: 10.4, frames: 1 }], 10);
  assert.equal(result.length, 5);
  assert.ok(Math.abs(result[4] - 0.3) < 1e-7);
  assert.equal(result[2], 0);
  assert.equal(result[3], 0);
  assert.throws(() => scheduledReplay(chunks, [{ startAt: 10, frames: 2 }], 10), /Incomplete/);
});

test("adaptive start holds exactly the predicted shortfall", () => {
  assert.equal(adaptiveStartSeconds({ rate: 1.5, receivedSeconds: 0.24, estimatedSeconds: 10 }), 0.24);
  // rate .6 discounted to .5: (1/.5 - 1)·(10 - 0) = 10 s of buffer plus the floor
  assert.ok(Math.abs(adaptiveStartSeconds({ rate: 0.6, receivedSeconds: 0, estimatedSeconds: 10 }) - 10.24) < 1e-9);
  assert.equal(adaptiveStartSeconds({ rate: 0, receivedSeconds: 0, estimatedSeconds: 1 }), Infinity);
  assert.ok(estimateSpeechSeconds("Guten Morgen.") > 1.04);
});

test("adaptive player starts on the first block when generation is fast, waits when slow", () => {
  let ms = 0;
  const now = () => ms;
  const fast = context();
  const pf = new StreamPlayer(fast, { sampleRate: 100, adaptive: { estimatedSeconds: 5, rateHint: 1.6 }, now });
  ms = 150;
  pf.play(pcm(24));
  assert.equal(fast.sources.length, 1);
  ms = 0;
  const slow = context();
  const ps = new StreamPlayer(slow, { sampleRate: 100, adaptive: { estimatedSeconds: 5, rateHint: 0.8 }, now });
  ms = 300;
  ps.play(pcm(24));
  assert.equal(slow.sources.length, 0);
  assert.ok(ps.requiredStartSeconds > 1);
  ps.finish();
  assert.equal(slow.sources.length, 1);
});
