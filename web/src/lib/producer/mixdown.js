// The producer's mixdown, in the Rust core (crates/wasm's arr_* functions over crates/core/src/arrange.rs). Takes are
// copied into the module once and stay there; each edit only resends the tracks.

import { takeOrder } from "./project.js";

let corePromise = null;
/** The core wasm module, instantiated once on the main thread (the voice worker has its own). */
export function loadCore() {
  corePromise ??= fetch("/wasm/driftscape.wasm")
    .then((r) => r.arrayBuffer())
    .then((bytes) => WebAssembly.instantiate(bytes, {}))
    .then(({ instance }) => instance.exports);
  return corePromise;
}

export class Mixdown {
  #core;
  #index = new Map(); // take id -> source index in the module

  constructor(core) {
    this.#core = core;
    core.arr_reset(32000);
  }

  /** Drop every take (a different project was opened). */
  clear() {
    this.#core.arr_reset(32000);
    this.#index.clear();
  }

  has(takeId) {
    return this.#index.has(takeId);
  }

  /** Copy a take's audio into the module. `channels` is an array of Float32Arrays. */
  load(takeId, channels, sampleRate) {
    if (this.#index.has(takeId)) return;
    const core = this.#core;
    const index = core.arr_source(channels[0].length, channels.length, sampleRate);
    if (index < 0) throw new Error("the core refused a take");
    channels.forEach((samples, c) => {
      // A fresh view after every call: the module's memory may have grown and moved.
      new Float32Array(core.memory.buffer, core.arr_channel(index, c), samples.length).set(samples);
    });
    this.#index.set(takeId, index);
  }

  /** Mix the project down to stereo at `sampleRate`. Every take it uses must be loaded. Returns [left, right]. */
  render(project, sampleRate) {
    const core = this.#core;
    core.arr_set_rate(sampleRate);
    core.arr_tracks_clear();
    const order = takeOrder(project);
    for (const id of order) if (!this.#index.has(id)) throw new Error(`take ${id} isn't loaded`);
    for (const t of project.tracks) {
      core.arr_track(t.gainDb, t.pan, t.mute ? 1 : 0);
      for (const c of t.clips) {
        core.arr_clip(this.#index.get(c.take), c.at, c.offset, c.length ?? -1, c.fadeIn, c.fadeOut, c.curve === "linear" ? 1 : 0, c.gainDb);
      }
    }
    const frames = Number(core.arr_render());
    if (frames < 0) throw new Error("the mixdown failed");
    return [0, 1].map((ch) => new Float32Array(core.memory.buffer, core.arr_output(ch), frames).slice());
  }
}

/** Stereo 24-bit WAV, what a DAW expects. */
export function stereoWav([left, right], sampleRate) {
  const frames = left.length;
  const bytes = new Uint8Array(44 + frames * 6);
  const view = new DataView(bytes.buffer);
  const text = (at, s) => [...s].forEach((ch, i) => view.setUint8(at + i, ch.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + frames * 6, true);
  text(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 6, true);
  view.setUint16(32, 6, true);
  view.setUint16(34, 24, true);
  text(36, "data");
  view.setUint32(40, frames * 6, true);
  let at = 44;
  for (let i = 0; i < frames; i++) {
    for (const s of [left[i], right[i]]) {
      const v = Math.round(Math.max(-1, Math.min(1, s)) * 8388607);
      bytes[at++] = v & 0xff;
      bytes[at++] = (v >> 8) & 0xff;
      bytes[at++] = (v >> 16) & 0xff;
    }
  }
  return new Blob([bytes], { type: "audio/wav" });
}
