// Gapless streaming playback with an adaptive, no-cut start.
//
// Written against the minimal Web Audio surface below, so it runs on a browser
// AudioContext and on React Native through react-native-audio-api (which
// implements the same API). Chunks are scheduled back-to-back as
// AudioBufferSourceNodes on the audio clock.

import { adaptiveStartSeconds } from "./audio.js";

export interface AudioBufferLike {
  readonly duration: number;
  copyToChannel(source: Float32Array, channelNumber: number): void;
}

export interface AudioBufferSourceLike {
  buffer: AudioBufferLike | null;
  connect(destination: unknown): unknown;
  start(when?: number): void;
  stop(when?: number): void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- bivariant with DOM Event handlers
  onended: ((event: any) => void) | null;
}

export interface AudioContextLike {
  readonly currentTime: number;
  readonly destination: unknown;
  createBuffer(channels: number, length: number, sampleRate: number): AudioBufferLike;
  createBufferSource(): AudioBufferSourceLike;
}

export interface AdaptiveStart {
  /** Expected total speech duration in seconds (estimateSpeechSeconds). */
  estimatedSeconds: number;
  /** Generation speed (audio s per wall s) from earlier runs, used before enough audio arrived. */
  rateHint?: number;
  margin?: number;
}

export interface StreamPlayerOptions {
  sampleRate?: number;
  /** Fixed audio to buffer before starting (seconds). */
  primeSeconds?: number;
  /** Scheduling lead when playback starts. */
  leadSeconds?: number;
  /** Hold everything until finish() (full-buffer mode). */
  waitForComplete?: boolean;
  /** Start as soon as the measured generation speed guarantees no cuts. */
  adaptive?: AdaptiveStart | null;
  /** Destination node; defaults to context.destination. */
  destination?: unknown;
  /** Wall clock in ms (injectable for tests). */
  now?: () => number;
}

export interface ScheduledEvent {
  arrivalAt: number;
  startAt: number;
  frames: number;
  isSilence: boolean;
  gapSeconds: number;
}

interface Pending {
  audio: Float32Array;
  arrivalAt: number;
  isSilence: boolean;
}

export class StreamPlayer {
  readonly context: AudioContextLike;
  readonly sampleRate: number;
  readonly primeSeconds: number;
  readonly leadSeconds: number;
  readonly waitForComplete: boolean;
  readonly adaptive: AdaptiveStart | null;
  /** Late chunks (> 20 ms behind schedule) and their total delay. */
  gaps = 0;
  gapSeconds = 0;
  events: ScheduledEvent[] = [];
  receivedSeconds = 0;
  /** Buffer the adaptive rule required at the last check. */
  requiredStartSeconds: number | null = null;
  finished = false;
  cancelled = false;
  readonly done: Promise<void>;

  private destination: unknown;
  private now: () => number;
  private startedMs: number;
  private pending: Pending[] = [];
  private queuedSeconds = 0;
  private sources = new Set<AudioBufferSourceLike>();
  private nextStart: number | null = null;
  private resolveDone!: () => void;

  constructor(context: AudioContextLike, options: StreamPlayerOptions = {}) {
    const perf = (globalThis as { performance?: { now(): number } }).performance;
    this.context = context;
    this.sampleRate = options.sampleRate ?? 24000;
    this.primeSeconds = options.primeSeconds ?? 0;
    this.leadSeconds = options.leadSeconds ?? 0.05;
    this.waitForComplete = options.waitForComplete ?? false;
    this.adaptive = options.adaptive ?? null;
    this.destination = options.destination ?? context.destination;
    this.now = options.now ?? (perf ? () => perf.now() : () => Date.now());
    this.startedMs = this.now();
    this.done = new Promise((resolve) => (this.resolveDone = resolve));
  }

  /** Audio-clock time playback started, or null while still buffering. */
  get startedAt(): number | null {
    return this.events.length ? this.events[0].startAt : null;
  }

  play(audio: Float32Array, meta: { isSilence?: boolean } = {}): void {
    if (this.cancelled || this.finished || !audio.length) return;
    const item = { audio, arrivalAt: this.context.currentTime, isSilence: !!meta.isSilence };
    this.receivedSeconds += audio.length / this.sampleRate;
    if (this.nextStart === null) {
      this.pending.push(item);
      this.queuedSeconds += audio.length / this.sampleRate;
      if (!this.waitForComplete && this.queuedSeconds >= this.startThreshold()) this.flush();
    } else this.schedule(item);
  }

  startThreshold(): number {
    if (!this.adaptive) return this.primeSeconds;
    const elapsed = (this.now() - this.startedMs) / 1000;
    const measured = elapsed > 0 ? this.receivedSeconds / elapsed : 0;
    // The first blocks include text conditioning, so lean on the hint until then.
    const rate = this.receivedSeconds < 0.7 && this.adaptive.rateHint ? this.adaptive.rateHint : measured;
    this.requiredStartSeconds = adaptiveStartSeconds({
      rate,
      receivedSeconds: this.receivedSeconds,
      estimatedSeconds: this.adaptive.estimatedSeconds,
      margin: this.adaptive.margin,
    });
    return Math.max(this.primeSeconds, this.requiredStartSeconds);
  }

  private flush(): void {
    if (this.cancelled || !this.pending.length) return;
    this.nextStart = this.context.currentTime + this.leadSeconds;
    for (const item of this.pending) this.schedule(item);
    this.pending = [];
    this.queuedSeconds = 0;
  }

  private schedule({ audio, arrivalAt, isSilence }: Pending): void {
    const ctx = this.context;
    const buffer = ctx.createBuffer(1, audio.length, this.sampleRate);
    buffer.copyToChannel(audio, 0);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.destination);
    const now = ctx.currentTime;
    const gap = Math.max(0, now - this.nextStart!);
    const startAt = Math.max(now, this.nextStart!);
    // Clock quantization below 20 ms is kept in the trace, not counted as a perceptible gap.
    if (gap > 0.02) {
      this.gaps++;
      this.gapSeconds += gap;
    }
    this.events.push({ arrivalAt, startAt, frames: audio.length, isSilence, gapSeconds: gap });
    this.nextStart = startAt + buffer.duration;
    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      this.checkDone();
    };
    source.start(startAt);
  }

  /** No more audio: release anything buffered; resolves when playback ends. */
  finish(): Promise<void> {
    this.flush();
    this.finished = true;
    this.checkDone();
    return this.done;
  }

  private checkDone(): void {
    if (this.finished && !this.sources.size) this.resolveDone();
  }

  stop(): void {
    this.cancelled = true;
    this.pending = [];
    this.finished = true;
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
    }
    this.sources.clear();
    this.checkDone();
  }
}

/** Reconstruct what the listener heard (late gaps as silence) from a playback trace. */
export function scheduledReplay(chunks: readonly Float32Array[], events: readonly Pick<ScheduledEvent, "startAt" | "frames">[], sampleRate: number): Float32Array {
  if (chunks.length !== events.length || !events.length) throw new Error("Incomplete playback trace");
  const first = events[0].startAt;
  const offsets = events.map((e) => Math.round((e.startAt - first) * sampleRate));
  const count = Math.max(...events.map((e, i) => offsets[i] + e.frames));
  const pcm = new Float32Array(count);
  chunks.forEach((c, i) => {
    if (c.length !== events[i].frames) throw new Error("Trace frame mismatch");
    pcm.set(c, offsets[i]);
  });
  return pcm;
}
