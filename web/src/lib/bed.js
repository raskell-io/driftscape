// Ambience beds in the browser: rain, wind, waves and fire from the Rust core (crates/core/src/ambience.rs), rendered
// as a seamless loop that an AudioBufferSourceNode repeats. The same generators run natively in the CLI and the plugin.

export const BEDS = [
  { id: "rain", name: "Rain" },
  { id: "wind", name: "Wind" },
  { id: "waves", name: "Waves" },
  { id: "fire", name: "Fire" },
];

/** A loop of `seconds` of the bed as planar stereo [left, right] (copied out of the module). */
export function bedLoop(core, id, sampleRate, seconds = 45, intensity = 0.6, seed = (Math.random() * 2 ** 32) >>> 0) {
  const index = BEDS.findIndex((b) => b.id === id);
  if (index < 0) throw new Error(`no bed "${id}"`);
  const frames = Number(core.bed_loop(index, sampleRate, seconds, seed, intensity));
  if (frames < 0) throw new Error("the core couldn't make that bed");
  return [0, 1].map((c) => new Float32Array(core.memory.buffer, core.bed_channel(c), frames).slice());
}

/** The bed a description calls for ("rain on a tin roof" → "rain"), or null. */
export function suggestBed(core, text) {
  const bytes = new TextEncoder().encode(text ?? "");
  new Uint8Array(core.memory.buffer, core.text(bytes.length), bytes.length).set(bytes);
  const i = core.bed_suggest();
  return i >= 0 ? BEDS[i].id : null;
}

/**
 * The bed a genre puts under the listener's words: `genre.bed` is a bed id, "auto" (from the words), or absent (none).
 * With `choice` "auto" the genre decides; "off" or a bed id overrides it.
 */
export function bedFor(core, genre, words, choice = "auto") {
  if (choice === "off") return null;
  if (choice !== "auto") return choice;
  if (!genre?.bed) return null;
  const text = words?.trim() ? words : genre.default_prompt;
  return genre.bed === "auto" ? suggestBed(core, text) : genre.bed;
}
