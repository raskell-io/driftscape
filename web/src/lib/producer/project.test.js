import { test } from "node:test";
import assert from "node:assert/strict";
import {
  appendTake,
  clipDuration,
  duplicateClip,
  extendClip,
  newProject,
  newTrack,
  projectSeconds,
  removeClip,
  removeTrack,
  renderFile,
  replaceTake,
  snap,
  takeOrder,
  trimClip,
} from "./project.js";

const info = (seconds, overlapSeconds = 0) => ({ seconds, prompt: "p", genreId: "ambient", overlapSeconds });

function setup() {
  const p = newProject();
  const t = newTrack({ genreId: "ambient" });
  p.tracks.push(t);
  return { p, t };
}

test("the first take starts at the playhead, the next crossfades into it", () => {
  const { p, t } = setup();
  const a = appendTake(p, t, "a", info(10), 4);
  assert.equal(a.at, 4);
  const b = appendTake(p, t, "b", info(10));
  assert.equal(b.at, 12, "starts the join fade before the first ends at 14");
  assert.equal(a.fadeOut, 2);
  assert.equal(b.fadeIn, 2);
  assert.equal(projectSeconds(p), 22);
});

test("an extension lines its replayed start up with the end of the take it continues", () => {
  const { p, t } = setup();
  const a = appendTake(p, t, "a", info(30), 0);
  trimClip(p, a, 20);
  a.offset = 2;
  const e = extendClip(p, t, a, "e", info(33, 3));
  // a plays take a from 2 s to its end (30 s) on the timeline from 0 to 28; the extension's first 3 s replay a's last
  // 3 s, so it starts at 25.
  assert.equal(a.length, null);
  assert.equal(e.at, 25);
  assert.equal(e.fadeIn, 3);
  assert.equal(a.fadeOut, 3);
  assert.equal(a.curve, "linear");
  assert.equal(t.clips[1], e);
});

test("duplicate, trim, snap and durations", () => {
  const { p, t } = setup();
  const a = appendTake(p, t, "a", info(8), 0);
  const d = duplicateClip(p, t, a);
  assert.equal(d.at, 8 - a.fadeOut);
  assert.equal(d.take, "a");
  trimClip(p, a, 100);
  assert.equal(clipDuration(a, 8), 8, "a trim can't outlast the take");
  trimClip(p, a, 0.4);
  assert.ok(a.fadeIn <= 0.2 && a.fadeOut <= 0.2, "fades shrink with the clip");
  assert.equal(snap(1.26), 1.5);
  assert.equal(snap(-3), 0);
  assert.equal(snap(1.1, 120), 1.0);
});

test("takes are forgotten once no clip plays them", () => {
  const { p, t } = setup();
  const a = appendTake(p, t, "a", info(8), 0);
  const d = duplicateClip(p, t, a);
  assert.equal(removeClip(p, t, a), null, "the duplicate still plays it");
  assert.equal(removeClip(p, t, d), "a");
  assert.equal(p.takes.a, undefined);

  const b = appendTake(p, t, "b", info(8), 0);
  assert.equal(replaceTake(p, b, "c", info(9)), "b");
  assert.deepEqual(Object.keys(p.takes), ["c"]);
  assert.deepEqual(removeTrack(p, t), ["c"]);
  assert.equal(p.tracks.length, 0);
});

test("the render file matches crates/core's arrangement format", () => {
  const { p, t } = setup();
  appendTake(p, t, "a", info(8), 0);
  appendTake(p, t, "b", info(8));
  const other = newTrack({ genreId: "ambient" });
  other.pan = -0.5;
  p.tracks.push(other);
  appendTake(p, other, "a", info(8), 1);
  assert.deepEqual(takeOrder(p), ["a", "b"]);
  const file = renderFile(p, 48000, (take) => `${take}.wav`);
  assert.deepEqual(file.sources, ["a.wav", "b.wav"]);
  assert.equal(file.tracks[0].clips[1].source, 1);
  assert.equal(file.tracks[1].clips[0].source, 0);
  assert.equal(file.tracks[1].pan, -0.5);
  assert.equal(file.tracks[0].clips[0].curve, "equal-power");
});
