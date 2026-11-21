// Live and radio mode in the browser: a port of crates/cli/src/live.rs onto Web Audio.
//
// A generator loop asks the voice for one segment at a time; each continues from the previous segment's tokens, so
// its start replays the previous one's end. A control loop schedules each transition ahead of time on the
// AudioContext clock: a crossfade into the next segment inside that overlap, or, if the next segment isn't ready, a
// crossfade back to the current one's rejoin point. A tweak drops the waiting segment, cancels the one being made,
// and asks for a short urgent segment from what is playing now, which comes in as soon as it's ready.
//
// Events (onEvent): { type: "segment", label } · { type: "repeat" } · { type: "progress", fraction } ·
//                   { type: "state", playing } · { type: "error", message }

import { CancelledError } from "./voice/voice.js";

const URGENT_SECONDS = 6;
const CHANGE_FADE = 3;
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
    this.recording = null; // Int16 chunks while recording
  }

  /** Start a session. Must be called from a user gesture (it creates the AudioContext). */
  async start({ genre, prompt = "", station = null }) {
    if (this.running) this.stop();
    this.ctx ??= new AudioContext({ latencyHint: "playback" });
    await this.ctx.resume();
    if (!this.output) {
      await this.ctx.audioWorklet.addModule("/worklets/output.js");
      this.output = new AudioWorkletNode(this.ctx, "driftscape-output", { outputChannelCount: [2] });
      this.output.port.onmessage = ({ data }) => this.#capture(data);
      this.bus = new GainNode(this.ctx);
      this.bus.connect(this.output).connect(this.ctx.destination);
    }
    this.params = { genre, prompt, focus: 0, variety: 0 };
    this.station = station;
    this.version++;
    this.ready = null;
    this.playingTokens = null;
    this.running = true;
    this.onEvent({ type: "state", playing: true });
    this.#generateLoop().catch((e) => this.#fail(e));
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

  /** Change what plays next: { prompt?, genre?, focus?, variety? }. */
  tweak(change) {
    if (!this.params) return;
    const before = JSON.stringify(this.params);
    this.params = { ...this.params, ...change };
    if (JSON.stringify(this.params) === before) return;
    // Start over from what is playing: drop the waiting segment, cancel the one being made.
    this.ready = null;
    this.version++;
    this.abort?.abort();
  }

  // ---- recording ---------------------------------------------------------------------------------------------------

  startRecording() {
    this.recording = [];
    this.output?.port.postMessage({ type: "record", on: true });
  }

  /** Stop recording and return it as a WAV Blob (mono, 16-bit, at the context's rate), or null. */
  stopRecording() {
    this.output?.port.postMessage({ type: "record", on: false });
    const chunks = this.recording;
    this.recording = null;
    if (!chunks?.length) return null;
    return wav(chunks, this.ctx.sampleRate);
  }

  recordedSeconds() {
    return this.recording ? this.recording.reduce((n, c) => n + c.length, 0) / this.ctx.sampleRate : 0;
  }

  #capture(samples) {
    if (!this.recording) return;
    const pcm = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) pcm[i] = Math.max(-1, Math.min(1, samples[i])) * 32767;
    this.recording.push(pcm);
  }

  // ---- generation --------------------------------------------------------------------------------------------------

  async #generateLoop() {
    let previous = null; // codebooks of the segment the next one continues
    let placedVersion = this.version;
    let gain = null;
    let number = 0;
    while (this.running) {
      const v = this.version;
      const { genre: g, prompt, focus, variety } = this.params;
      const listener = this.station
        ? this.station.prompts[Math.floor(number / Math.max(1, this.station.segments_per_prompt)) % this.station.prompts.length]
        : prompt;
      const urgent = v !== placedVersion || number === 0;
      const seconds = urgent ? Math.min(g.segment_seconds, URGENT_SECONDS) : g.segment_seconds;
      const frames = Math.round(g.overlap_seconds * 50);
      const request = {
        prompt: fill(g, listener),
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
        urgent,
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
    const at = end - fade;
    if (now + LOOKAHEAD < at) return;
    this.#fadeOut(p, at, fade);
    if (this.ready) {
      // The new segment's (overlap - fade) sounds like the current one's (end - fade).
      const segment = this.#take();
      this.#play(segment, at, Math.max(0, segment.overlap - fade), fade);
    } else {
      this.onEvent({ type: "repeat" });
      this.#play(p.segment, at, p.segment.rejoin, fade, true);
    }
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
    const voice = { segment, source, gain, start: at, offset };
    this.voices = [...(this.voices ?? []).filter((v) => !v.done), voice];
    source.onended = () => (voice.done = true);
    this.playing = voice;
    if (!repeat) this.onEvent({ type: "segment", label: segment.label });
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

/** The genre's template with the listener's words (driftscape_core::preset::Genre::prompt). */
export function fill(genre, listener) {
  const words = listener.trim() || genre.default_prompt;
  return genre.template.replace("{prompt}", words).replace("{bpm}", genre.bpm ? genre.bpm.toFixed(0) : "");
}

const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Mono 16-bit WAV from Int16 chunks. */
function wav(chunks, sampleRate) {
  const frames = chunks.reduce((n, c) => n + c.length, 0);
  const header = new DataView(new ArrayBuffer(44));
  const text = (at, s) => [...s].forEach((ch, i) => header.setUint8(at + i, ch.charCodeAt(0)));
  text(0, "RIFF");
  header.setUint32(4, 36 + frames * 2, true);
  text(8, "WAVEfmt ");
  header.setUint32(16, 16, true);
  header.setUint16(20, 1, true);
  header.setUint16(22, 1, true);
  header.setUint32(24, sampleRate, true);
  header.setUint32(28, sampleRate * 2, true);
  header.setUint16(32, 2, true);
  header.setUint16(34, 16, true);
  text(36, "data");
  header.setUint32(40, frames * 2, true);
  return new Blob([header, ...chunks], { type: "audio/wav" });
}
