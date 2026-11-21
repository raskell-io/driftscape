import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { BEDS, bedFor, bedLoop, suggestBed } from "./bed.js";

const wasm = new URL("../../static/wasm/driftscape.wasm", import.meta.url);

test("beds come from the Rust core: loops, suggestions, genre rules", { skip: !existsSync(wasm) && "run prepare-assets first" }, async () => {
  const { instance } = await WebAssembly.instantiate(readFileSync(wasm), {});
  const core = instance.exports;
  for (const { id } of BEDS) {
    const [l, r] = bedLoop(core, id, 16000, 8, 0.6, 3);
    assert.equal(l.length, 8 * 16000);
    assert.equal(r.length, l.length);
    const rms = Math.sqrt(l.reduce((s, x) => s + x * x, 0) / l.length);
    assert.ok(rms > 0.01 && rms < 0.3, `${id} rms ${rms}`);
    assert.ok(l.some((x, i) => x !== r[i]), `${id} is stereo`);
  }
  assert.equal(suggestBed(core, "Rain on a tin roof"), "rain");
  assert.equal(suggestBed(core, "a campfire by the lake"), "fire");
  assert.equal(suggestBed(core, "soft piano"), null);

  const realistic = { bed: "auto", default_prompt: "steady rain on leaves" };
  const ambient = { default_prompt: "warm pads" };
  assert.equal(bedFor(core, realistic, ""), "rain", "the genre's default words");
  assert.equal(bedFor(core, realistic, "harbour at dusk"), "waves");
  assert.equal(bedFor(core, ambient, "rain on the window"), null, "no bed unless the genre has one");
  assert.equal(bedFor(core, ambient, "anything", "fire"), "fire", "a chosen bed wins");
  assert.equal(bedFor(core, realistic, "rain", "off"), null);
});
