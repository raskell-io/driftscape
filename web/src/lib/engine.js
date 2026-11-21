// Live and radio mode in the browser: a port of crates/cli/src/live.rs onto Web Audio.
//
// A generator loop asks the voice for one segment at a time; each continues from the previous segment's tokens, so
// its start replays the previous one's end. A control loop schedules each transition ahead of time on the
// AudioContext clock: a crossfade into the next segment inside that overlap, or, if the next segment isn't ready, a
// crossfade back to the current one's rejoin point. A tweak drops the waiting segment, cancels the one being made,
// and asks for a short urgent segment from what is playing now, which comes in as soon as it's ready.
//
// Events (onEvent): { type: "segment", label } · { type: "repeat" } · { type: "progress", fraction } ·
//                   { type: "state", playing } · { type: "error", message } ·
//                   { type: "snapshot", snapshot } when a new segment starts (see store.js; the page saves it so the
//                   session can resume later)

import { CancelledError } from "./voice/voice.js";
import { appendRecording, finishRecording, startRecording } from "./store.js";

const URGENT_SECONDS = 6;
const CHANGE_FADE = 3;
// Repeats crossfade longer than transitions, so going back sounds like a phrase returning.
const REPEAT_FADE = 4;
const START_FADE = 0.5;
// Background tabs run timers about once a second, so transitions are scheduled well ahead.
const LOOKAHEAD = 1.5;
const TICK_MS = 100;
// Per-segment loudness, as in live.rs; the output worklet levels what remains.
const TARGET_RMS = 0.1;
const MAX_GAIN = 8;
const PEAK_LIMIT = 0.95;

/** Equal-power fade curves (sin rising, cos falling), sampled for setValueCurveAtTime. */
const CURVE_POINTS = 128;
const FADE_IN = Float32Array.from({ length: CURVE_POINTS }, (_, i) => Math.sin(((i / (CURVE_POINTS - 1)) * Math.PI) / 2));
const FADE_OUT = Float32Array.from(FADE_IN, (_, i) => FADE_IN[CURVE_POINTS - 1 - i]);

export class LiveEngine {
  /**
   * @param {any} voice a WorkerVoice or NativeVoice
   * @param {(event: any) => void} onEvent
   */
  constructor(voice, onEvent) {
    this.voice = voice;
    this.onEvent = onEvent;
    this.ctx = null;
    this.running = false;
    this.params = null; // { genre, prompt, focus, variety }
    this.station = null;
    this.version = 0;
    this.ready = null; // segment waiting to play
    this.playingTokens = null;
    this.playing = null; // { segment, buffer, start, offset, gain, source }
    this.abort = null;
    this.recording = null; // { id, pending: Int16Array[], pendingFrames, frames } while recording
  }

  /**
   * Start a session. Must be called from a user gesture (it creates the AudioContext). With `resume` (a snapshot from
   * store.js) the snapshot's segment plays at once and generation continues from its tokens.
   */
  async start({ genre, prompt = "", station = null, focus = 0, variety = 0, blend = null, resume = null }) {
    if (this.running) this.stop();
    this.ctx ??= new AudioContext({ latencyHint: "playback" });
    await this.ctx.resume();
    if (!this.output) {
      await this.ctx.audioWorklet.addModule("/worklets/output.js");
      this.output = new AudioWorkletNode(this.ctx, "driftscape-output", { outputChannelCount: [2] });
      this.output.port.onmessage = ({ data }) => this.#capture(data);
      this.bus = new GainNode(this.ctx, { gain: this.level ?? 1 });
      this.bus.connect(this.output).connect(this.ctx.destination);
      // For the visuals: how loud the output is right now.
      this.analyser = new AnalyserNode(this.ctx, { fftSize: 1024 });
      this.output.connect(this.analyser);
      this.scope = new Float32Array(this.analyser.fftSize);
    }
    this.params = { genre, prompt, focus, variety, blend };
    this.station = station;
    this.version++;
    this.ready = null;
    this.playingTokens = null;
    this.running = true;
    let start = { previous: null, number: 0 };
    if (resume) {
      const buffer = new AudioBuffer({ length: resume.audio.length, sampleRate: resume.sampleRate, numberOfChannels: 1 });
      buffer.copyToChannel(resume.audio, 0);
      this.ready = { ...resume, buffer, urgent: false, speed: null, label: `${resume.label} · resumed` };
      start = { previous: resume.codebooks, number: resume.number };
    }
    this.segmentsMade = start.number;
    this.onEvent({ type: "state", playing: true });
    this.#generateLoop(start).catch((e) => this.#fail(e));
    this.timer = setInterval(() => this.#tick(), TICK_MS);
  }

  stop() {
    this.running = false;
    clearInterval(this.timer);
    this.abort?.abort();
    if (this.playing && this.ctx) {
      const now = this.ctx.currentTime;
      this.playing.gain.gain.cancelScheduledValues(now);
      this.playing.gain.gain.setTargetAtTime(0, now, 0.3);
      this.playing.source.stop(now + 2);
    }
    for (const v of this.voices ?? []) v.source.stop(this.ctx.currentTime + 2);
    this.voices = [];
    this.playing = null;
    this.ready = null;
    this.onEvent({ type: "state", playing: false });
  }

  /**
   * Change what plays: { prompt?, genre?, blend?, focus?, variety? }. `blend` is a second genre mixed in
   * ({ genre, share }), from the terrain pad.
   *
   * An urgent tweak (the default: the listener asked for something) drops the waiting segment, cancels the one being
   * made, and brings the change in within seconds. A gentle one (the terrain's autopilot drifting) only changes what
   * the next segment is made from, so the music evolves without interruptions.
   */
  tweak(change, { urgent = true } = {}) {
    if (!this.params) return;
    const before = JSON.stringify(this.params);
    this.params = { ...this.params, ...change };
    if (JSON.stringify(this.params) === before || !urgent) return;
    this.ready = null;
    this.version++;
    this.abort?.abort();
  }

  /** The output's loudness right now (RMS, roughly 0..0.4), for the visuals. */
  loudness() {
    if (!this.analyser || !this.running) return 0;
    this.analyser.getFloatTimeDomainData(this.scope);
    let sum = 0;
    for (const v of this.scope) sum += v * v;
    return Math.sqrt(sum / this.scope.length);
  }

  /** Master level, 0..1. */
  setLevel(level) {
    this.level = level;
    if (this.bus) this.bus.gain.setTargetAtTime(level, this.ctx.currentTime, 0.1);
  }

  // ---- recording ---------------------------------------------------------------------------------------------------
  // Recordings go to IndexedDB in chunks of a couple of seconds as they are made (store.js), so a reload or a crash
  // loses at most the last chunk.

  async startRecording(label) {
    const sampleRate = this.ctx?.sampleRate ?? 48000;
    const id = await startRecording({ sampleRate, label });
    this.recording = { id, pending: [], pendingFrames: 0, frames: 0, sampleRate };
    this.output?.port.postMessage({ type: "record", on: true });
    return id;
  }

  /** Stop recording; resolves with the recording's id once everything is stored. */
  async stopRecording() {
    this.output?.port.postMessage({ type: "record", on: false });
    const rec = this.recording;
    this.recording = null;
    if (!rec) return null;
    await this.#flush(rec);
    await finishRecording(rec.id);
    return rec.id;
  }

  recordedSeconds() {
    return this.recording ? this.recording.frames / this.recording.sampleRate : 0;
  }

  #capture(samples) {
    const rec = this.recording;
    if (!rec) return;
    const pcm = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) pcm[i] = Math.max(-1, Math.min(1, samples[i])) * 32767;
    rec.pending.push(pcm);
    rec.pendingFrames += pcm.length;
    rec.frames += pcm.length;
    if (rec.pendingFrames >= rec.sampleRate * 2) this.#flush(rec);
  }

  #flush(rec) {
    if (!rec.pendingFrames) return Promise.resolve();
    const chunk = new Int16Array(rec.pendingFrames);
    let at = 0;
    for (const p of rec.pending) {
      chunk.set(p, at);
      at += p.length;
    }
    rec.pending = [];
    rec.pendingFrames = 0;
    // Chunks are written in order: IndexedDB transactions on the same stores run in the order they were created.
    return appendRecording(rec.id, chunk).catch((e) => this.onEvent({ type: "error", message: `saving the recording: ${e.message}` }));
  }

  // ---- generation --------------------------------------------------------------------------------------------------

  async #generateLoop({ previous, number }) {
    // previous: codebooks of the segment the next one continues; number: segments made so far.
    let placedVersion = this.version;
    let gain = null;
    while (this.running) {
      const v = this.version;
      const { genre: g, prompt, focus, variety, blend } = this.params;
      const listener = this.station
        ? this.station.prompts[Math.floor(number / Math.max(1, this.station.segments_per_prompt)) % this.station.prompts.length]
        : prompt;
      const urgent = v !== placedVersion || (number === 0 && !this.ready);
      // A short urgent segment gets music (or a change) to the listener sooner, but only pays off when generation keeps
      // up with playback. When it doesn't (in the browser, about half real time), a short segment just means repeating
      // one short loop while the next is made; a full-length one gives longer phrases to come back to.
      const keepsUp = (this.speed ?? (this.voice.fast ? 1.5 : 0.5)) >= 0.9;
      const seconds = urgent && keepsUp ? Math.min(g.segment_seconds, URGENT_SECONDS) : g.segment_seconds;
      const frames = Math.round(g.overlap_seconds * 50);
      const request = {
        prompt: fill(g, listener, blend),
        seconds,
        guidance: clamp(g.guidance + focus, 1, 8),
        temperature: clamp(g.temperature + variety, 0.5, 1.6),
        topK: 250,
        seed: Math.floor(Math.random() * 2 ** 52),
        continueFrom: previous ? previous.map((c) => c.slice(-frames)) : null,
        analysis: { crossfadeSeconds: g.crossfade_seconds, bpm: g.bpm },
      };
      this.abort = new AbortController();
      let result;
      try {
        result = await this.voice.generate(request, (f) => this.onEvent({ type: "progress", fraction: f }), this.abort.signal);
      } catch (e) {
        if (e instanceof CancelledError) {
          previous = this.playingTokens;
          continue;
        }
        throw e;
      }
      if (!this.running) return;
      // How fast this voice generates, for sizing urgent segments (seconds of new audio per second).
      const measured = seconds / result.seconds;
      this.speed = this.speed === undefined ? measured : 0.5 * this.speed + 0.5 * measured;
      if (this.version !== v) {
        previous = this.playingTokens;
        continue;
      }

      // Loudness: steady level, changing gradually between segments, never clipping.
      const audio = result.audio;
      let sum = 0;
      let peak = 1e-6;
      for (const s of audio) {
        sum += s * s;
        peak = Math.max(peak, Math.abs(s));
      }
      const rms = Math.sqrt(sum / audio.length);
      const wanted = Math.min(TARGET_RMS / Math.max(rms, 1e-6), MAX_GAIN);
      gain = Math.min(gain === null ? wanted : 0.5 * gain + 0.5 * wanted, PEAK_LIMIT / peak);
      const buffer = new AudioBuffer({ length: audio.length, sampleRate: result.sampleRate, numberOfChannels: 1 });
      buffer.copyToChannel(audio.map((s) => s * gain), 0);

      const segment = {
        buffer,
        codebooks: result.codebooks,
        overlap: result.overlapSeconds,
        crossfade: Math.min(g.crossfade_seconds, Math.max(result.overlapSeconds, 1)),
        rejoin: result.rejoin / result.sampleRate,
        rejoins: (result.rejoins ?? [result.rejoin]).map((r) => r / result.sampleRate),
        urgent,
        speed: seconds / result.seconds, // seconds of new audio per second of generation
        label: [
          g.name,
          listener.trim() || g.default_prompt,
          g.bpm ? (result.bpm ? `${result.bpm.toFixed(1)} BPM` : "no pulse") : null,
          `${(seconds / result.seconds).toFixed(1)}× real time`,
        ]
          .filter(Boolean)
          .join(" · "),
      };

      // Wait for the slot; a tweak meanwhile makes this segment stale.
      while (this.running && this.ready && this.version === v) await sleep(50);
      if (!this.running) return;
      if (this.version !== v) {
        previous = this.playingTokens;
        continue;
      }
      this.ready = segment;
      previous = segment.codebooks;
      placedVersion = v;
      number++;
      this.segmentsMade = number;
    }
  }

  // ---- control -----------------------------------------------------------------------------------------------------

  #tick() {
    if (!this.running) return;
    const now = this.ctx.currentTime;
    const p = this.playing;
    if (!p) {
      if (this.ready) this.#play(this.#take(), now + 0.1, 0, START_FADE);
      return;
    }
    const end = p.start + (p.segment.buffer.duration - p.offset);
    // A change the listener asked for comes in now, once the current segment's own fade-in is over.
    if (this.ready?.urgent && now + 0.2 >= p.start + p.segment.crossfade && now + 0.2 + CHANGE_FADE < end) {
      const at = now + 0.2;
      const segment = this.#take();
      this.#fadeOut(p, at, CHANGE_FADE);
      this.#play(segment, at, Math.max(0, segment.overlap - CHANGE_FADE), CHANGE_FADE);
      return;
    }
    const fade = p.segment.crossfade;
    if (this.ready) {
      const at = end - fade;
      if (now + LOOKAHEAD < at) return;
      // The new segment's (overlap - fade) sounds like the current one's (end - fade).
      const segment = this.#take();
      this.#fadeOut(p, at, fade);
      this.#play(segment, at, Math.max(0, segment.overlap - fade), fade);
      return;
    }
    // Nothing new yet: repeat, going back to the next of the segment's rejoin points in turn. A rejoin point sounds like
    // the segment's last `fade` seconds, so the new voice must reach it when the old one reaches them (at end - fade).
    // The repeat's crossfade is longer (REPEAT_FADE) and so starts earlier, but never longer than the audio that comes
    // before the rejoin point, and never before the current voice has finished fading in.
    const rejoins = p.segment.rejoins?.length ? p.segment.rejoins : [p.segment.rejoin];
    const rejoin = rejoins[(p.repeats ?? 0) % rejoins.length];
    const fadeLen = Math.max(fade, Math.min(REPEAT_FADE, rejoin + fade, end - fade - (p.start + p.fadeIn)));
    const rat = end - fadeLen;
    if (now + LOOKAHEAD < rat) return;
    const offset = Math.max(0, rejoin - (fadeLen - fade));
    this.onEvent({ type: "repeat" });
    if (trace()) {
      console.debug(
        `[driftscape] repeat at ${rat.toFixed(2)} (now ${now.toFixed(2)}, voice started ${p.start.toFixed(2)}): back to ${rejoin.toFixed(1)} s of ${p.segment.buffer.duration.toFixed(1)} s, loop ${(p.segment.buffer.duration - fade - rejoin).toFixed(1)} s, crossfade ${fadeLen.toFixed(1)} s`,
      );
    }
    this.#fadeOut(p, rat, fadeLen);
    this.#play(p.segment, rat, offset, fadeLen, true);
    this.playing.repeats = (p.repeats ?? 0) + 1;
  }

  #take() {
    const segment = this.ready;
    this.ready = null;
    this.playingTokens = segment.codebooks;
    return segment;
  }

  #play(segment, at, offset, fade, repeat = false) {
    const source = new AudioBufferSourceNode(this.ctx, { buffer: segment.buffer });
    const gain = new GainNode(this.ctx, { gain: 0 });
    source.connect(gain).connect(this.bus);
    gain.gain.setValueCurveAtTime(FADE_IN, at, fade);
    source.start(at, offset);
    const voice = { segment, source, gain, start: at, offset, fadeIn: fade };
    this.voices = [...(this.voices ?? []).filter((v) => !v.done), voice];
    source.onended = () => (voice.done = true);
    this.playing = voice;
    if (!repeat) {
      this.onEvent({ type: "segment", label: segment.label, speed: segment.speed });
      this.onEvent({ type: "snapshot", snapshot: this.#snapshot(segment) });
    }
  }

  /** What store.js keeps to resume this session from `segment`. */
  #snapshot(segment) {
    const { genre, prompt, focus, variety, blend } = this.params;
    return {
      audio: segment.buffer.getChannelData(0).slice(),
      sampleRate: segment.buffer.sampleRate,
      codebooks: segment.codebooks,
      overlap: segment.overlap,
      crossfade: segment.crossfade,
      rejoin: segment.rejoin,
      rejoins: segment.rejoins,
      // The genre and prompt, without this session's speed or a previous "resumed".
      label: segment.label.replace(/ · resumed$/, "").replace(/ · [\d.]+× real time/, ""),
      params: { genreId: genre.id, prompt, focus, variety, blendId: blend?.genre.id ?? null, blendShare: blend?.share ?? 0 },
      stationId: this.station?.id ?? null,
      number: this.segmentsMade ?? 0,
      savedAt: Date.now(),
    };
  }

  #fadeOut(voice, at, fade) {
    voice.gain.gain.setValueCurveAtTime(FADE_OUT, at, fade);
    voice.source.stop(at + fade);
  }

  #fail(e) {
    this.onEvent({ type: "error", message: e.message ?? String(e) });
    this.stop();
  }
}

/**
 * The genre's template with the listener's words (driftscape_core::preset::Genre::prompt), and a second genre's
 * flavor when the terrain pad sits between two regions.
 */
export function fill(genre, listener, blend = null) {
  const words = listener.trim() || genre.default_prompt;
  const base = genre.template.replace("{prompt}", words).replace("{bpm}", genre.bpm ? genre.bpm.toFixed(0) : "");
  if (!blend || blend.share < 0.25) return base;
  return `${base}, ${blend.share > 0.4 ? "blended with" : "with a touch of"} ${blend.genre.flavor}`;
}

const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
/** Debug output, on when localStorage has "driftscape-trace". */
const trace = () => {
  try {
    return !!globalThis.localStorage?.getItem("driftscape-trace");
  } catch {
    return false;
  }
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
