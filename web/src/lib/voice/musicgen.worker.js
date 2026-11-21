// MusicGen (small) in the browser, on onnxruntime-web with WebGPU. A port of crates/infer/src/musicgen: the same ONNX
// files, the same delay pattern, classifier-free guidance, top-k sampling, continuation with a batched replay of the
// overlap, and cancellation. After generating, it finds the segment's rejoin point and tempo with the Rust core
// (static/wasm/driftscape.wasm), so the live engine needs no heavy work on the main thread.
//
// Messages in:  { type: "load" }
//               { type: "generate", id, request }   (see voice.js for the request and result shapes)
//               { type: "cancel", id }
// Messages out: { type: "loading", detail, fraction } · { type: "ready" } · { type: "error", id?, message }
//               { type: "progress", id, step, total } · { type: "result", id, ... } · { type: "cancelled", id }

import * as ort from "onnxruntime-web/webgpu";
import { Tokenizer } from "./tokenizer.js";

const REPO = "https://huggingface.co/Xenova/musicgen-small/resolve/main/";
const FILES = {
  tokenizer: "tokenizer.json",
  text: "onnx/text_encoder_q4.onnx",
  decoder: "onnx/decoder_model_merged_q4.onnx",
  codec: "onnx/encodec_decode_q4.onnx",
};
const MODEL_CACHE = "driftscape-models-v1";

const CODEBOOKS = 4;
const VOCAB = 2048;
const PAD = 2048n;
const LAYERS = 24;
const HEADS = 16;
const HEAD_DIM = 64;
const TEXT_DIM = 768;
const FRAME_RATE = 50;
const SAMPLE_RATE = 32000;
const MAX_SECONDS = 30;

let model = null; // { tokenizer, text, decoder, codec, core }
let current = null; // id of the generation in progress
const cancelled = new Set();

// ---- loading -------------------------------------------------------------------------------------------------------

/** Run a step, naming it in any error so a failure says where it happened. */
async function step(name, f) {
  try {
    return await f();
  } catch (e) {
    throw new Error(`${name}: ${e.message ?? e}`);
  }
}

/** Fetch a file through the Cache API, reporting progress as bytes arrive. */
async function fetchCached(url, onBytes) {
  const cache = await caches.open(MODEL_CACHE);
  const hit = await cache.match(url);
  if (hit) {
    const buf = await hit.arrayBuffer();
    onBytes(buf.byteLength, buf.byteLength);
    return buf;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`downloading ${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onBytes(received, total);
  }
  const buf = new Uint8Array(received);
  let at = 0;
  for (const c of chunks) {
    buf.set(c, at);
    at += c.byteLength;
  }
  await cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
  return buf.buffer;
}

async function loadRuntime() {
  // The 28 MB runtime ships gzipped (Cloudflare Pages caps files at 25 MiB); unpack it here.
  // Some servers mark .gz files as Content-Encoding: gzip, so the browser has already unpacked them; look at the bytes
  // instead of trusting the name: gzip starts 1f 8b, wasm starts \0asm.
  const res = await fetch("/ort/ort-wasm-simd-threaded.asyncify.wasm.gz");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const bytes = await res.arrayBuffer();
  const head = new Uint8Array(bytes, 0, 2);
  const gzipped = head[0] === 0x1f && head[1] === 0x8b;
  ort.env.wasm.wasmBinary = gzipped
    ? await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer()
    : bytes;
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = 1; // the GPU does the work; threads would need cross-origin isolation
}

async function loadCore() {
  const bytes = await (await fetch("/wasm/driftscape.wasm")).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

async function load() {
  if (!("gpu" in navigator) || !(await navigator.gpu.requestAdapter())) {
    throw new Error("this browser has no WebGPU; try a recent Chrome, Edge or Safari, or run `driftscape serve`");
  }
  post({ type: "loading", detail: "Preparing the engine", fraction: 0 });
  const [, core] = await Promise.all([step("loading the ONNX runtime", loadRuntime), step("loading the core", loadCore)]);

  // Download everything first (in parallel), with one combined progress figure.
  const names = Object.keys(FILES);
  const progress = Object.fromEntries(names.map((n) => [n, { got: 0, total: 0 }]));
  const report = () => {
    const got = names.reduce((s, n) => s + progress[n].got, 0);
    const total = names.reduce((s, n) => s + progress[n].total, 0);
    if (total) post({ type: "loading", detail: "Downloading the model", fraction: got / total });
  };
  const buffers = await Promise.all(
    names.map((n) =>
      step(`downloading ${FILES[n]}`, () =>
        fetchCached(REPO + FILES[n], (got, total) => {
          progress[n] = { got, total };
          report();
        }),
      ),
    ),
  );
  const files = Object.fromEntries(names.map((n, i) => [n, buffers[i]]));

  post({ type: "loading", detail: "Starting the model on your GPU", fraction: 1 });
  const tokenizer = new Tokenizer(JSON.parse(new TextDecoder().decode(files.tokenizer)));
  const options = { executionProviders: ["webgpu"] };
  // The decoder's caches stay on the GPU between steps; only the logits come back.
  const cacheOutputs = {};
  for (let l = 0; l < LAYERS; l++)
    for (const part of ["decoder.key", "decoder.value", "encoder.key", "encoder.value"])
      cacheOutputs[`present.${l}.${part}`] = "gpu-buffer";
  // One at a time: onnxruntime-web refuses to create two WebGPU sessions concurrently.
  const [text, decoder, codec] = await step("starting the model", async () => [
    await ort.InferenceSession.create(files.text, options),
    await ort.InferenceSession.create(files.decoder, { ...options, preferredOutputLocation: cacheOutputs }),
    await ort.InferenceSession.create(files.codec, options),
  ]);
  model = { tokenizer, text, decoder, codec, core };

  // Warm up: the first run on WebGPU compiles the GPU programs, which can take a long time. Do it now, during loading,
  // so pressing play doesn't pay for it: a fresh clip, then a continuation of it (the batched replay uses other shapes).
  post({ type: "loading", detail: "Warming up the engine", fraction: 1 });
  const warm = { prompt: "ambient pads", seconds: 0.4, guidance: 3, temperature: 1, topK: 50, seed: 1, continueFrom: null };
  const first = await step("warming up", () => generate(-1, warm));
  await step("warming up", () => generate(-1, { ...warm, continueFrom: first.codebooks.map((c) => c.slice(-10)) }));
}

// ---- generation ----------------------------------------------------------------------------------------------------

/** SplitMix64, seeded per request, so a request is reproducible. */
function rng(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  return () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    z ^= z >> 31n;
    return Number(z >> 11n) / 2 ** 53;
  };
}

/** Top-k sampling with temperature (crates/infer/src/sampling.rs). */
function sample(logits, topK, temperature, random) {
  const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, Math.max(1, topK));
  const max = logits[idx[0]];
  const t = Math.max(temperature, 1e-4);
  const weights = idx.map((i) => Math.exp((logits[i] - max) / t));
  let target = random() * weights.reduce((a, b) => a + b, 0);
  for (let j = 0; j < idx.length; j++) {
    target -= weights[j];
    if (target <= 0) return idx[j];
  }
  return idx[idx.length - 1];
}

/** Whether codebook k must hold the pad token at position p of a len-position sequence. */
const forcedPad = (k, p, len) => p <= k || p >= len - CODEBOOKS + 1 + k;

/** The token codebook k must take at position p, or null when the model chooses (continuation-aware). */
function forcedToken(k, p, len, prefix) {
  if (forcedPad(k, p, len)) return PAD;
  const frame = p - k - 1;
  return prefix && frame < prefix[k].length ? BigInt(prefix[k][frame]) : null;
}

const i64 = (values, dims) => new ort.Tensor("int64", BigInt64Array.from(values, (v) => BigInt(v)), dims);

async function generate(id, req) {
  const { tokenizer, text, decoder, codec } = model;
  const prefix = req.continueFrom ?? null; // number[CODEBOOKS][frames]
  const prefixFrames = prefix ? prefix[0].length : 0;
  const newFrames = Math.round(req.seconds * FRAME_RATE);
  const frames = prefixFrames + newFrames;
  if (newFrames <= 0 || frames > MAX_SECONDS * FRAME_RATE) throw new Error(`MusicGen makes up to ${MAX_SECONDS} s`);
  const len = frames + CODEBOOKS;
  const steps = len - 1;

  // Text → hidden states, then the conditional and (zeroed) unconditional batches side by side.
  const ids = tokenizer.encode(req.prompt);
  const seq = ids.length;
  const enc = await text.run({ input_ids: i64(ids, [1, seq]), attention_mask: i64(ids.map(() => 1), [1, seq]) });
  const hidden = enc.last_hidden_state.data;
  const states = new Float32Array(2 * seq * TEXT_DIM);
  states.set(hidden);
  const mask = new BigInt64Array(2 * seq).fill(1n, 0, seq);
  const statesT = new ort.Tensor("float32", states, [2, seq, TEXT_DIM]);
  const maskT = new ort.Tensor("int64", mask, [2, seq]);

  const tokens = Array.from({ length: CODEBOOKS }, () => new Array(len).fill(PAD));
  const random = rng(req.seed);
  const input = (q) => {
    const row = tokens.map((c) => c[q]);
    return [...row, ...row]; // both batches take the same tokens
  };
  let cache = null; // { "past_key_values.*": Tensor }

  async function run(positions) {
    const n = positions.length;
    // input_ids is [rows, positions]: transpose the per-position rows.
    const flat = new BigInt64Array(2 * CODEBOOKS * n);
    for (let t = 0; t < n; t++) for (let r = 0; r < 2 * CODEBOOKS; r++) flat[r * n + t] = positions[t][r];
    const feeds = {
      input_ids: new ort.Tensor("int64", flat, [2 * CODEBOOKS, n]),
      encoder_hidden_states: statesT,
      encoder_attention_mask: maskT,
      use_cache_branch: new ort.Tensor("bool", [cache !== null], [1]),
    };
    for (let l = 0; l < LAYERS; l++)
      for (const part of ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]) {
        const name = `past_key_values.${l}.${part}`;
        feeds[name] = cache ? cache[name] : new ort.Tensor("float32", new Float32Array(0), [2, HEADS, 0, HEAD_DIM]);
      }
    const out = await decoder.run(feeds);
    // Self-attention caches grow each step; cross-attention caches come from the first pass only (on cached passes
    // the merged decoder returns placeholders for them).
    const next = {};
    for (let l = 0; l < LAYERS; l++)
      for (const part of ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]) {
        const past = `past_key_values.${l}.${part}`;
        const present = out[`present.${l}.${part}`];
        if (cache && part.startsWith("encoder")) {
          next[past] = cache[past];
          present.dispose();
        } else {
          next[past] = present;
          if (cache) cache[past].dispose();
        }
      }
    cache = next;
    // The last position's logits, [rows][VOCAB].
    const logits = out.logits;
    const data = await logits.getData();
    const positionsOut = logits.dims[1];
    const rows = [];
    for (let r = 0; r < 2 * CODEBOOKS; r++) {
      const at = (r * positionsOut + positionsOut - 1) * VOCAB;
      rows.push(data.subarray(at, at + VOCAB));
    }
    return rows;
  }

  try {
    // Every leading position whose tokens are all forced (the start token and a continuation's overlap) goes in as
    // one batch.
    let known = 1;
    while (known < len && [...Array(CODEBOOKS).keys()].every((k) => forcedToken(k, known, len, prefix) !== null)) known++;
    for (let q = 1; q < known; q++) for (let k = 0; k < CODEBOOKS; k++) tokens[k][q] = forcedToken(k, q, len, prefix);
    let logits = await run([...Array(known).keys()].map(input));
    const guided = new Float32Array(VOCAB);
    for (let p = known; p <= steps; p++) {
      if (cancelled.has(id)) throw new Cancelled();
      if (p > known) logits = await run([input(p - 1)]);
      for (let k = 0; k < CODEBOOKS; k++) {
        const forced = forcedToken(k, p, len, prefix);
        if (forced !== null) {
          tokens[k][p] = forced;
          continue;
        }
        const cond = logits[k];
        const uncond = logits[CODEBOOKS + k];
        for (let v = 0; v < VOCAB; v++) guided[v] = uncond[v] + req.guidance * (cond[v] - uncond[v]);
        tokens[k][p] = BigInt(sample(guided, req.topK, req.temperature, random));
      }
      if (p % 10 === 0 || p === steps) post({ type: "progress", id, step: p, total: steps });
    }
  } finally {
    if (cache) for (const t of Object.values(cache)) t.dispose();
  }

  // Undo the stagger: codebook k's frames sit at positions k+1 ..= k+frames.
  const codebooks = tokens.map((c, k) => c.slice(k + 1, k + 1 + frames).map(Number));
  const codes = new BigInt64Array(CODEBOOKS * frames);
  codebooks.forEach((c, k) => c.forEach((t, f) => (codes[k * frames + f] = BigInt(t))));
  const audioOut = await codec.run({ audio_codes: new ort.Tensor("int64", codes, [1, 1, CODEBOOKS, frames]) });
  const audio = new Float32Array(audioOut.audio_values.data);
  return { audio, codebooks, overlapSeconds: prefixFrames / FRAME_RATE };
}

class Cancelled extends Error {}

/** Rejoin point and tempo, with the Rust core (crates/wasm). */
function analyse(audio, overlap, analysis) {
  const { core } = model;
  const ptr = core.buffer(audio.length);
  new Float32Array(core.memory.buffer, ptr, audio.length).set(audio);
  const crossfade = Math.min(Math.round(analysis.crossfadeSeconds * SAMPLE_RATE), audio.length);
  // Compare a short window at the seam: enough to match the sound, cheap enough for wasm.
  const window = Math.min(crossfade, SAMPLE_RATE / 2);
  const from = audio.length - crossfade;
  const latest = Math.max(0, from - 4 * SAMPLE_RATE); // repeats at least 4 s long
  const bar = analysis.bpm ? Math.round(((4 * 60) / analysis.bpm) * SAMPLE_RATE) : 0;
  let rejoin = Number(core.rejoin(from, window, Math.min(overlap, latest), latest, bar));
  if (rejoin < 0) rejoin = Number(core.rejoin(from, window, 0, latest, 0));
  const bpm = analysis.bpm ? core.tempo(SAMPLE_RATE) : 0;
  return { rejoin: rejoin < 0 ? overlap : rejoin, bpm };
}

// ---- messages ------------------------------------------------------------------------------------------------------

function post(message, transfer = []) {
  self.postMessage(message, transfer);
}

self.onmessage = async ({ data }) => {
  if (data.type === "load") {
    try {
      await load();
      post({ type: "ready" });
    } catch (e) {
      post({ type: "error", message: e.message ?? String(e) });
    }
  } else if (data.type === "cancel") {
    if (current === data.id) cancelled.add(data.id);
  } else if (data.type === "generate") {
    const { id, request } = data;
    current = id;
    const started = performance.now();
    try {
      const { audio, codebooks, overlapSeconds } = await generate(id, request);
      const overlap = Math.round(overlapSeconds * SAMPLE_RATE);
      const { rejoin, bpm } = analyse(audio, overlap, request.analysis);
      post(
        {
          type: "result",
          id,
          audio,
          sampleRate: SAMPLE_RATE,
          codebooks,
          overlapSeconds,
          rejoin,
          bpm,
          seconds: (performance.now() - started) / 1000,
        },
        [audio.buffer],
      );
    } catch (e) {
      if (e instanceof Cancelled) post({ type: "cancelled", id });
      else post({ type: "error", id, message: e.message ?? String(e) });
    } finally {
      cancelled.delete(id);
      current = null;
    }
  }
};
