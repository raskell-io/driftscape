// What Driftscape keeps on this device, in IndexedDB (nothing leaves it):
//   settings    the last mode, genre, prompt, knobs and station, restored on the next visit
//   session     a snapshot of the segment playing now (its audio and tokens), so a session can resume where it left
//               off: the snapshot plays at once and generation continues from its tokens
//   recordings  recordings, written in chunks while they are made, so a reload or a crash loses nothing
//
// The model itself is in Cache Storage (the voice worker's "driftscape-models-v1"); see storage() below.

const DB = "driftscape";
const VERSION = 1;

let dbPromise = null;
function db() {
  dbPromise ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, VERSION);
    req.onupgradeneeded = () => {
      const d = req.result;
      d.createObjectStore("kv"); // settings, session
      d.createObjectStore("recordings", { keyPath: "id" });
      const chunks = d.createObjectStore("chunks", { keyPath: ["recording", "seq"] });
      chunks.createIndex("recording", "recording");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

/** Run `f` in a transaction and resolve with its request's result once the transaction commits. */
async function tx(stores, mode, f) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const t = d.transaction(stores, mode);
    let result;
    const r = f(t);
    if (r && "onsuccess" in r) r.onsuccess = () => (result = r.result);
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

const get = (key) => tx("kv", "readonly", (t) => t.objectStore("kv").get(key));
const put = (key, value) => tx("kv", "readwrite", (t) => t.objectStore("kv").put(value, key));

// ---- settings ----------------------------------------------------------------------------------------------------

export const loadSettings = () => get("settings").catch(() => undefined);
export const saveSettings = (settings) => put("settings", settings).catch(() => {});

// ---- session -----------------------------------------------------------------------------------------------------

/**
 * @typedef {object} Snapshot
 * @property {Float32Array} audio   the segment's samples (loudness already applied)
 * @property {number} sampleRate
 * @property {number[][]} codebooks
 * @property {number} overlap       seconds at the start that replay the previous segment
 * @property {number} crossfade     seconds
 * @property {number} rejoin        seconds
 * @property {string} label
 * @property {object} params        { genreId, prompt, focus, variety }
 * @property {string|null} stationId
 * @property {number} number        how many segments the session had made
 * @property {number} savedAt       ms since the epoch
 */

export const loadSession = () => get("session").catch(() => undefined);
export const saveSession = (snapshot) => put("session", snapshot).catch(() => {});
export const clearSession = () => put("session", undefined).catch(() => {});

// ---- recordings --------------------------------------------------------------------------------------------------

/** Start a recording; returns its id. */
export async function startRecording({ sampleRate, label }) {
  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  await tx("recordings", "readwrite", (t) =>
    t.objectStore("recordings").put({ id, sampleRate, label, startedAt: Date.now(), frames: 0, chunks: 0, done: false }),
  );
  return id;
}

/** Append 16-bit samples to a recording. */
export async function appendRecording(id, samples) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const t = d.transaction(["recordings", "chunks"], "readwrite");
    const recordings = t.objectStore("recordings");
    const req = recordings.get(id);
    req.onsuccess = () => {
      const rec = req.result;
      if (!rec) return;
      t.objectStore("chunks").put({ recording: id, seq: rec.chunks, data: samples });
      rec.chunks += 1;
      rec.frames += samples.length;
      recordings.put(rec);
    };
    t.oncomplete = () => resolve();
    t.onerror = () => reject(t.error);
  });
}

export async function finishRecording(id) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const t = d.transaction("recordings", "readwrite");
    const s = t.objectStore("recordings");
    const req = s.get(id);
    req.onsuccess = () => req.result && s.put({ ...req.result, done: true, endedAt: Date.now() });
    t.oncomplete = () => resolve();
    t.onerror = () => reject(t.error);
  });
}

/** All recordings, newest first. One left `done: false` was interrupted (a reload or a crash); it is still complete up
 * to its last chunk. */
export async function listRecordings() {
  const all = (await tx("recordings", "readonly", (t) => t.objectStore("recordings").getAll())) ?? [];
  return all.sort((a, b) => b.startedAt - a.startedAt);
}

/** A recording as a mono 16-bit WAV Blob. */
export async function recordingWav(id) {
  const rec = await tx("recordings", "readonly", (t) => t.objectStore("recordings").get(id));
  const chunks = await tx("chunks", "readonly", (t) => t.objectStore("chunks").index("recording").getAll(id));
  chunks.sort((a, b) => a.seq - b.seq);
  return wav(
    chunks.map((c) => c.data),
    rec.sampleRate,
  );
}

export async function deleteRecording(id) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const t = d.transaction(["recordings", "chunks"], "readwrite");
    t.objectStore("recordings").delete(id);
    t.objectStore("chunks").delete(IDBKeyRange.bound([id, 0], [id, Number.MAX_SAFE_INTEGER]));
    t.oncomplete = () => resolve();
    t.onerror = () => reject(t.error);
  });
}

/** Mono 16-bit WAV from Int16 chunks. */
export function wav(chunks, sampleRate) {
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

// ---- storage -----------------------------------------------------------------------------------------------------

export const MODEL_CACHE = "driftscape-models-v1";

/** Ask the browser not to evict our data (the model is a 570 MB download). Returns whether it is persistent. */
export async function persist() {
  if (!navigator.storage?.persist) return false;
  return (await navigator.storage.persisted()) || (await navigator.storage.persist());
}

/** { used, quota, persisted, modelCached } in bytes. */
export async function storage() {
  const est = (await navigator.storage?.estimate?.()) ?? {};
  const persisted = (await navigator.storage?.persisted?.()) ?? false;
  const modelCached = (await caches.has(MODEL_CACHE)) && (await (await caches.open(MODEL_CACHE)).keys()).length >= 4;
  return { used: est.usage ?? 0, quota: est.quota ?? 0, persisted, modelCached };
}

/** Delete the downloaded model; the next start downloads it again. */
export const removeModel = () => caches.delete(MODEL_CACHE);
