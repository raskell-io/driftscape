import { test } from "node:test";
import assert from "node:assert/strict";
import { blendAt, contour, differs, regions } from "./terrain.js";

const genres = ["ambient", "synthetic-ambience", "realistic-ambience", "ambient-techno", "thematic"].map((id) => ({ id }));
const rs = regions(genres);

test("on a region's peak, that genre plays alone", () => {
  const r = rs.find((r) => r.genre.id === "ambient-techno");
  const b = blendAt(rs, r.x, r.y);
  assert.equal(b.genre.id, "ambient-techno");
  assert.equal(b.blend, null);
});

test("halfway between two regions, they blend evenly", () => {
  const a = rs.find((r) => r.genre.id === "ambient");
  const t = rs.find((r) => r.genre.id === "thematic");
  const b = blendAt(rs, (a.x + t.x) / 2, (a.y + t.y) / 2);
  assert.ok(["ambient", "thematic"].includes(b.genre.id));
  assert.ok(b.blend && b.blend.share > 0.4, JSON.stringify(b));
});

test("small moves don't count as changes; crossing into another region does", () => {
  const a = blendAt(rs, 0.36, 0.5);
  assert.equal(differs(a, blendAt(rs, 0.37, 0.51)), false);
  assert.equal(differs(a, blendAt(rs, 0.76, 0.75)), true);
});

test("contours of a single bump form a closed ring", () => {
  const gx = 20, gy = 20, field = new Float32Array(21 * 21);
  for (let j = 0; j <= gy; j++) for (let i = 0; i <= gx; i++) field[j * 21 + i] = Math.exp(-((i / gx - 0.5) ** 2 + (j / gy - 0.5) ** 2) / 0.05);
  const segs = contour(field, gx, gy, 0.5);
  assert.ok(segs.length > 10);
  // every point lies near the same radius from the centre
  const radii = segs.map(([x, y]) => Math.hypot(x - 0.5, y - 0.5));
  assert.ok(Math.max(...radii) - Math.min(...radii) < 0.03);
});
