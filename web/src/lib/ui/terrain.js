// The terrain behind the Live instrument: each genre is a hill on a map, the listener is a dot, and where the dot is
// decides the music. Pure functions, so they can be tested without a canvas.

/** Where each genre sits on the map (0..1 in both directions) and its colour. Unknown genres go round a circle. */
const LAYOUT = {
  ambient: { x: 0.36, y: 0.5, color: [255, 176, 143] },
  "synthetic-ambience": { x: 0.77, y: 0.28, color: [169, 147, 240] },
  "realistic-ambience": { x: 0.17, y: 0.22, color: [255, 211, 161] },
  "ambient-techno": { x: 0.76, y: 0.75, color: [228, 139, 176] },
  thematic: { x: 0.37, y: 0.85, color: [176, 111, 176] },
};

/** Lay out regions for the given genres. */
export function regions(genres) {
  return genres.map((genre, i) => {
    const known = LAYOUT[genre.id];
    if (known) return { genre, ...known };
    const a = (i / genres.length) * Math.PI * 2;
    return { genre, x: 0.5 + 0.32 * Math.cos(a), y: 0.5 + 0.32 * Math.sin(a), color: [233, 220, 207] };
  });
}

const SPREAD = 0.03;

/**
 * The genre mix at a point: the dominant genre, and a second one with its share (0..1 of the two), or null when the
 * second is too faint to matter.
 */
export function blendAt(rs, x, y) {
  const w = rs.map((r) => {
    const dx = x - r.x;
    const dy = y - r.y;
    return Math.exp(-(dx * dx + dy * dy) / SPREAD);
  });
  const order = w.map((v, i) => i).sort((a, b) => w[b] - w[a]);
  const [first, second] = order;
  const pair = w[first] + w[second] || 1;
  const share = w[second] / pair;
  return {
    genre: rs[first].genre,
    blend: share >= 0.15 ? { genre: rs[second].genre, share: Math.round(share * 100) / 100 } : null,
  };
}

/** Whether two blends differ enough to make different music. */
export function differs(a, b) {
  if (!a || !b) return a !== b;
  if (a.genre.id !== b.genre.id) return true;
  const sa = a.blend?.share ?? 0;
  const sb = b.blend?.share ?? 0;
  if ((a.blend?.genre.id ?? null) !== (b.blend?.genre.id ?? null) && Math.max(sa, sb) >= 0.25) return true;
  return Math.abs(sa - sb) >= 0.15;
}

/** Terrain height at a point: a hill per region, a smaller one per landmark, and slow waves over time. */
export function height(rs, landmarks, x, y, t) {
  let h = 0;
  for (const r of rs) {
    const dx = x - r.x;
    const dy = (y - r.y) * 1.3;
    h += Math.exp(-(dx * dx + dy * dy) / 0.035);
  }
  for (const l of landmarks) {
    const dx = x - l.x;
    const dy = y - l.y;
    h += 0.6 * Math.exp(-(dx * dx + dy * dy) / 0.006);
  }
  return h + 0.08 * Math.sin(x * 9 + t * 0.4) * Math.cos(y * 7 - t * 0.3);
}

/**
 * Contour lines of a sampled field (marching squares). `field` is (gx+1)×(gy+1) samples, row-major. Returns line
 * segments in 0..1 coordinates as [x1, y1, x2, y2] for the given level.
 */
export function contour(field, gx, gy, level) {
  const segs = [];
  const at = (i, j) => field[j * (gx + 1) + i];
  for (let j = 0; j < gy; j++) {
    for (let i = 0; i < gx; i++) {
      const a = at(i, j);
      const b = at(i + 1, j);
      const c = at(i + 1, j + 1);
      const d = at(i, j + 1);
      const idx = (a > level) | ((b > level) << 1) | ((c > level) << 2) | ((d > level) << 3);
      if (idx === 0 || idx === 15) continue;
      const f = (p, q) => (level - p) / (q - p);
      const top = [(i + f(a, b)) / gx, j / gy];
      const right = [(i + 1) / gx, (j + f(b, c)) / gy];
      const bottom = [(i + f(d, c)) / gx, (j + 1) / gy];
      const left = [i / gx, (j + f(a, d)) / gy];
      const cases = {
        1: [left, top], 2: [top, right], 3: [left, right], 4: [right, bottom], 5: [left, top, right, bottom],
        6: [top, bottom], 7: [left, bottom], 8: [left, bottom], 9: [top, bottom], 10: [left, bottom, top, right],
        11: [right, bottom], 12: [left, right], 13: [top, right], 14: [left, top],
      }[idx];
      for (let s = 0; s < cases.length; s += 2) segs.push([cases[s][0], cases[s][1], cases[s + 1][0], cases[s + 1][1]]);
    }
  }
  return segs;
}
