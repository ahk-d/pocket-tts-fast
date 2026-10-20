// Ported from browser-quant-v1/test_metrics.mjs and test_latency.mjs.
import test from "node:test";
import assert from "node:assert/strict";
import { makeRecord, summarize, wavFromChunks, startLatency, simulateLivePlayback, decodeWav } from "../dist/index.js";

test("full generation RTF comes from elapsed time and returned sample duration", () => {
  const r = makeRecord({ elapsedMs: 1000, firstChunkMs: 200, sampleCount: 96000, sampleRate: 24000 });
  assert.equal(r.audioSeconds, 4);
  assert.equal(r.generationSeconds, 1);
  assert.equal(r.rtf, 0.25);
  assert.equal(r.firstChunkSeconds, 0.2);
  assert.throws(() => makeRecord({ elapsedMs: 1, sampleCount: 0, sampleRate: 24000 }));
});

test("summary excludes explicit warmup and failures while exposing their counts", () => {
  const r = summarize([{ warmup: true, rtf: 99, firstChunkSeconds: 99 }, { rtf: 0.5, firstChunkSeconds: 0.3 }, { rtf: 0.1, firstChunkSeconds: 0.1 }, { error: "failed", rtf: 0 }]);
  assert.equal(r.n, 2);
  assert.equal(r.medianRtf, 0.3);
  assert.equal(r.medianFirstChunkSeconds, 0.2);
  assert.equal(r.errors, 1);
  assert.equal(r.warmups, 1);
});

test("WAV duration and PCM encoding follow actual finite chunks", () => {
  const { buffer, sampleCount } = wavFromChunks([new Float32Array([0, 0.5]), new Float32Array([-0.5, 1])], 24000);
  const v = new DataView(buffer);
  assert.equal(sampleCount, 4);
  assert.equal(v.getUint32(24, true), 24000);
  assert.equal(v.getUint32(40, true), 8);
  assert.equal(v.getInt16(46, true), 16384);
  assert.throws(() => wavFromChunks([new Float32Array([NaN])], 24000));
  assert.throws(() => wavFromChunks([], 24000));
  assert.throws(() => wavFromChunks([new Float32Array(4)], 24000), /Silent/);
});

test("WAV decode round-trips 16-bit PCM and averages channels", () => {
  const { buffer } = wavFromChunks([new Float32Array([0, 0.5, -0.5, 1])], 24000);
  const { audio, sampleRate } = decodeWav(buffer);
  assert.equal(sampleRate, 24000);
  assert.deepEqual([...audio].map((x) => Math.round(x * 1000) / 1000), [0, 0.5, -0.5, 1]);
  // Stereo float32: one frame of (0.2, 0.4) averages to 0.3.
  const b = new ArrayBuffer(44 + 8);
  const v = new DataView(b);
  const tag = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  tag(0, "RIFF"); v.setUint32(4, 44, true); tag(8, "WAVE"); tag(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 3, true); v.setUint16(22, 2, true); v.setUint32(24, 48000, true);
  v.setUint32(28, 48000 * 8, true); v.setUint16(32, 8, true); v.setUint16(34, 32, true); tag(36, "data"); v.setUint32(40, 8, true);
  v.setFloat32(44, 0.2, true); v.setFloat32(48, 0.4, true);
  const st = decodeWav(b);
  assert.equal(st.sampleRate, 48000);
  assert.ok(Math.abs(st.audio[0] - 0.3) < 1e-6);
});

test("one-second target includes preparation and request setup, not just synthesis", () => {
  const r = startLatency({ requestStartedMs: 100, generationStartedMs: 900, snapshotPerformanceMs: 1200, snapshotAudioSeconds: 10.3, scheduledStartSeconds: 10.4 });
  assert.ok(Math.abs(r.requestToPlaybackSeconds - 1.2) < 1e-8);
  assert.equal(r.underOneSecond, false);
});

test("warm start uses both clocks without counting unrelated audio-context age", () => {
  const r = startLatency({ requestStartedMs: 10000, generationStartedMs: 10020, snapshotPerformanceMs: 10420, snapshotAudioSeconds: 9000.4, scheduledStartSeconds: 9000.45 });
  assert.ok(Math.abs(r.requestToPlaybackSeconds - 0.47) < 1e-8);
  assert.equal(r.underOneSecond, true);
});

test("invalid clock boundaries never create a false latency pass", () => {
  assert.throws(() => startLatency({ requestStartedMs: 10, generationStartedMs: 5, snapshotPerformanceMs: 6, snapshotAudioSeconds: 2, scheduledStartSeconds: 3 }), /clock/i);
  assert.throws(() => startLatency({ requestStartedMs: 0, generationStartedMs: 1, snapshotPerformanceMs: 2, snapshotAudioSeconds: 2, scheduledStartSeconds: NaN }), /clock/i);
});

test("a newly resumed audio clock cannot falsely pass the first-request target", () => {
  const r = startLatency({ requestStartedMs: 1000, generationStartedMs: 1040, snapshotPerformanceMs: 1998, snapshotAudioSeconds: 10.54, scheduledStartSeconds: 10.59 });
  assert.ok(Math.abs(r.requestToPlaybackSeconds - 1.048) < 1e-8);
  assert.ok(Math.abs(r.playbackStartSeconds - 1.008) < 1e-8);
  assert.equal(r.underOneSecond, false);
});

test("simulated live playback counts only late arrivals as gaps", () => {
  assert.deepEqual(simulateLivePlayback([{ seconds: 0.24, arrival: 0.3 }, { seconds: 0.24, arrival: 0.5 }]), { gaps: 0, gapSeconds: 0 });
  const late = simulateLivePlayback([{ seconds: 0.24, arrival: 0.3 }, { seconds: 0.24, arrival: 0.9 }]);
  assert.equal(late.gaps, 1);
  assert.ok(Math.abs(late.gapSeconds - 0.36) < 1e-9);
});
