// Producer mode's project: layers (tracks) of clips placed on a timeline. A clip plays a stretch of a take (generated
// audio, kept in IndexedDB). This mirrors crates/core/src/arrange.rs; the mixdown itself runs in Rust (mixdown.js),
// so the preview and the exported WAV are the same computation as `driftscape render`.
//
// Everything here is plain data and pure functions, tested with node --test.

/** Seconds of crossfade between a track's clips when a new take is placed after the last one. */
export const JOIN_FADE = 2;
/** Fade in and out at a fresh clip's ends. */
export const EDGE_FADE = 0.5;

let counter = 0;
const id = (prefix) => `${prefix}-${Date.now().toString(36)}-${(counter++).toString(36)}`;

/**
 * @typedef {{ seconds: number, prompt: string, genreId: string, overlapSeconds: number }} TakeInfo
 * @typedef {{ id: string, take: string, at: number, offset: number, length: number | null, fadeIn: number,
 *   fadeOut: number, curve: "equal-power" | "linear", gainDb: number }} Clip
 * @typedef {{ id: string, name: string, genreId: string, prompt: string, gainDb: number, pan: number, mute: boolean,
 *   clips: Clip[] }} Track
 * @typedef {{ id: string, name: string, tracks: Track[], takes: Record<string, TakeInfo> }} Project
 */

/** @returns {Project} */
export function newProject(name = "Untitled") {
  return { id: id("project"), name, tracks: [], takes: {} };
}

/** @returns {Track} */
export function newTrack({ genreId, prompt = "", name = "" }) {
  return { id: id("track"), name, genreId, prompt, gainDb: 0, pan: 0, mute: false, clips: [] };
}

/** @returns {Clip} */
export function newClip(take, at, fields = {}) {
  return { id: id("clip"), take, at, offset: 0, length: null, fadeIn: EDGE_FADE, fadeOut: EDGE_FADE, curve: "equal-power", gainDb: 0, ...fields };
}

/** How long a clip plays, given its take's length (as `Placed::duration` in Rust). */
export function clipDuration(clip, takeSeconds) {
  const rest = Math.max(0, takeSeconds - clip.offset);
  return clip.length == null ? rest : Math.min(Math.max(0, clip.length), rest);
}

export const clipEnd = (project, clip) => clip.at + clipDuration(clip, project.takes[clip.take]?.seconds ?? 0);

/** Where a track's last clip ends (0 for an empty track). */
export const trackEnd = (project, track) => Math.max(0, ...track.clips.map((c) => clipEnd(project, c)));

/** Where the whole piece ends. */
export const projectSeconds = (project) => Math.max(0, ...project.tracks.map((t) => trackEnd(project, t)));

/** Snap a time to the grid: a beat when the track's genre has a tempo, else half a second. Never before 0. */
export function snap(seconds, bpm = null) {
  const grid = bpm ? 60 / bpm : 0.5;
  return Math.max(0, Math.round(seconds / grid) * grid);
}

/**
 * Place a new take at the end of a track, crossfading into the last clip. Returns the new clip (already added).
 * The first clip on a track starts at `at` (the playhead) instead.
 */
export function appendTake(project, track, takeId, info, at = 0) {
  project.takes[takeId] = info;
  const last = lastClip(project, track);
  let clip;
  if (!last) {
    clip = newClip(takeId, at);
  } else {
    const fade = Math.min(JOIN_FADE, info.seconds / 2, clipDuration(last, project.takes[last.take].seconds) / 2);
    last.fadeOut = Math.max(last.fadeOut, fade);
    clip = newClip(takeId, Math.max(0, clipEnd(project, last) - fade), { fadeIn: fade });
  }
  track.clips.push(clip);
  return clip;
}

/**
 * Place an extension of `clip`: a take that continues its take's ending (its first `info.overlapSeconds` replay that
 * ending). The original clip plays to its take's end, and the two crossfade linearly over the replayed part, where
 * they line up. Returns the new clip.
 */
export function extendClip(project, track, clip, takeId, info) {
  project.takes[takeId] = info;
  const source = project.takes[clip.take];
  clip.length = null;
  const overlap = Math.min(info.overlapSeconds, clipDuration(clip, source.seconds));
  const at = clip.at + (source.seconds - clip.offset) - info.overlapSeconds;
  // Both fades share the clip's curve; an extended clip's fades become linear.
  clip.curve = "linear";
  clip.fadeOut = overlap;
  const next = newClip(takeId, at, { curve: "linear", fadeIn: overlap });
  track.clips.splice(track.clips.indexOf(clip) + 1, 0, next);
  return next;
}

/** A copy of a clip placed right after it. */
export function duplicateClip(project, track, clip) {
  const copy = { ...clip, id: id("clip"), at: clipEnd(project, clip) - clip.fadeOut };
  track.clips.splice(track.clips.indexOf(clip) + 1, 0, copy);
  return copy;
}

/** Remove a clip; also forget its take if nothing else plays it. Returns the take's id when it became unused. */
export function removeClip(project, track, clip) {
  track.clips = track.clips.filter((c) => c !== clip);
  return forgetIfUnused(project, clip.take);
}

/** Remove a track and the takes only it used. Returns the ids of takes that became unused. */
export function removeTrack(project, track) {
  project.tracks = project.tracks.filter((t) => t !== track);
  return [...new Set(track.clips.map((c) => c.take))].filter((take) => forgetIfUnused(project, take));
}

/** Point a clip at a new take of the same length class (regenerate). Returns the old take's id if it became unused. */
export function replaceTake(project, clip, takeId, info) {
  const old = clip.take;
  project.takes[takeId] = info;
  clip.take = takeId;
  clip.offset = Math.min(clip.offset, Math.max(0, info.seconds - 1));
  return forgetIfUnused(project, old);
}

function forgetIfUnused(project, takeId) {
  if (project.tracks.some((t) => t.clips.some((c) => c.take === takeId))) return null;
  delete project.takes[takeId];
  return takeId;
}

function lastClip(project, track) {
  let last = null;
  for (const c of track.clips) if (!last || clipEnd(project, c) > clipEnd(project, last)) last = c;
  return last;
}

/** Trim a clip's end to `length` seconds (at least a tenth of a second, at most what the take holds). */
export function trimClip(project, clip, length) {
  const rest = project.takes[clip.take].seconds - clip.offset;
  clip.length = Math.min(Math.max(0.1, length), rest);
  const d = clipDuration(clip, project.takes[clip.take].seconds);
  clip.fadeIn = Math.min(clip.fadeIn, d / 2);
  clip.fadeOut = Math.min(clip.fadeOut, d / 2);
}

/** The take ids a project uses, in a stable order: the mixdown's source indices. */
export function takeOrder(project) {
  const seen = [];
  for (const t of project.tracks) for (const c of t.clips) if (!seen.includes(c.take)) seen.push(c.take);
  return seen;
}

/** A project file for `driftscape render`: the arrangement plus WAV paths for its takes, named by `fileOf(takeId)`. */
export function renderFile(project, sampleRate, fileOf) {
  const order = takeOrder(project);
  return {
    sample_rate: sampleRate,
    sources: order.map(fileOf),
    tracks: project.tracks.map((t) => ({
      name: t.name,
      gain_db: t.gainDb,
      pan: t.pan,
      mute: t.mute,
      clips: t.clips.map((c) => ({
        source: order.indexOf(c.take),
        at: c.at,
        offset: c.offset,
        length: c.length,
        fade_in: c.fadeIn,
        fade_out: c.fadeOut,
        curve: c.curve,
        gain_db: c.gainDb,
      })),
    })),
  };
}
