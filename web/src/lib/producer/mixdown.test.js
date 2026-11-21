import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { Mixdown, stereoWav } from "./mixdown.js";
import { appendTake, newProject, newTrack } from "./project.js";

const wasm = new URL("../../../static/wasm/driftscape.wasm", import.meta.url);

test("the wasm mixdown places, fades, pans and resamples takes", { skip: !existsSync(wasm) && "run prepare-assets first" }, async () => {
  const { instance } = await WebAssembly.instantiate(readFileSync(wasm), {});
  const mix = new Mixdown(instance.exports);
  const rate = 32000;
  const tone = (seconds, f) => Float32Array.from({ length: seconds * rate }, (_, i) => 0.2 * Math.sin((2 * Math.PI * f * i) / rate));

  const p = newProject();
  const t = newTrack({ genreId: "ambient" });
  t.pan = 1;
  p.tracks.push(t);
  appendTake(p, t, "a", { seconds: 4, prompt: "", genreId: "ambient", overlapSeconds: 0 }, 1);
  appendTake(p, t, "b", { seconds: 4, prompt: "", genreId: "ambient", overlapSeconds: 0 });
  mix.load("a", [tone(4, 220)], rate);
  mix.load("b", [tone(4, 330)], rate);

  const [l, r] = mix.render(p, rate);
  assert.equal(l.length, 7 * rate, "1 s lead-in, 4 s, then 4 s starting 2 s before the end");
  assert.ok(l.every((s) => Math.abs(s) < 1e-6), "panned hard right");
  assert.equal(r[rate - 1], 0, "silent before the first clip");
  assert.ok(Math.max(...r.subarray(2 * rate, 3 * rate)) > 0.19);

  const [l48] = mix.render(p, 48000);
  assert.equal(l48.length, 7 * 48000, "the same takes, mixed at another rate");

  const blob = stereoWav([l, r], rate);
  assert.equal(blob.size, 44 + 7 * rate * 6);
  assert.throws(() => mix.render({ ...p, tracks: [{ ...t, clips: [{ ...t.clips[0], take: "missing" }] }] }, rate));
});
