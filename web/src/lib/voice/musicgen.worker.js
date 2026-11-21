// MusicGen (small) in the browser, on onnxruntime-web: on the GPU through WebGPU where the browser's WebGPU works with
// it, otherwise on the CPU (multithreaded when the page is cross-origin isolated). Firefox takes the CPU path:
// onnxruntime-web's WebGPU build fails there with a wasm memory fault. A port of crates/infer/src/musicgen: the same ONNX
// files, the same delay pattern, classifier-free guidance, top-k sampling, continuation with a batched replay of the
// overlap, and cancellation. After generating, it finds the segment's rejoin point and tempo with the Rust core
// (static/wasm/driftscape.wasm), so the live engine needs no heavy work on the main thread.
//
// Messages in:  { type: "load", backend: "webgpu" | "webgpu-js" | "cpu", precision: "q4" | "f16" }
//               { type: "generate", id, request }   (see voice.js for the request and result shapes)
//               { type: "cancel", id }
// Messages out: { type: "loading", detail, fraction } · { type: "ready" } · { type: "error", id?, message }
//               { type: "progress", id, step, total } · { type: "result", id, ... } · { type: "cancelled", id }

let ort; // the onnxruntime-web build for the chosen backend, imported in load()
let backend = "webgpu";
let precision = "q4";
let skipCheck = false; // diagnostics only: load even if the music decoder check fails
import { Tokenizer } from "./tokenizer.js";
// Greedy MusicGen output for a fixed prompt from the native engine (crates/infer/examples/greedy.rs --json): the tokens
// and the decoded audio's level and zero-crossing rate. The self-check at load compares against it.
import REFERENCE from "./selfcheck.reference.json";

const REPO = "https://huggingface.co/Xenova/musicgen-small/resolve/main/";
// q4: 4-bit weights, float32 arithmetic, one merged decoder. f16: the same 4-bit weights with float16 arithmetic (faster
// on GPUs with shader-f16), as two decoders: a first pass that takes many positions, and a cached step.
const FILE_SETS = {
  q4: {
    tokenizer: "tokenizer.json",
    text: "onnx/text_encoder_q4.onnx",
    decoder: "onnx/decoder_model_merged_q4.onnx",
    codec: "onnx/encodec_decode_q4.onnx",
  },
  f16: {
    tokenizer: "tokenizer.json",
    text: "onnx/text_encoder_q4f16.onnx",
    decoder: "onnx/decoder_model_q4f16.onnx",
    decoderPast: "onnx/decoder_with_past_model_q4f16.onnx",
    codec: "onnx/encodec_decode_q4f16.onnx",
  },
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
  // Each runtime ships gzipped (the WebGPU one is 28 MB; Cloudflare Pages caps files at 25 MiB); unpack it here.
  // Some servers mark .gz files as Content-Encoding: gzip, so the browser has already unpacked them; look at the bytes
  // instead of trusting the name: gzip starts 1f 8b, wasm starts \0asm.
  // webgpu: the native WebGPU implementation; webgpu-js: the JavaScript one (the package's default build); cpu: no GPU.
  const builds = {
    webgpu: ["ort-wasm-simd-threaded.asyncify", () => import("onnxruntime-web/webgpu")],
    "webgpu-js": ["ort-wasm-simd-threaded.jsep", () => import("onnxruntime-web")],
    cpu: ["ort-wasm-simd-threaded", () => import("onnxruntime-web/wasm")],
  };
  const [file, importer] = builds[backend];
  ort = await importer();
  const res = await fetch(`/ort/${file}.wasm.gz`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const bytes = await res.arrayBuffer();
  const head = new Uint8Array(bytes, 0, 2);
  const gzipped = head[0] === 0x1f && head[1] === 0x8b;
  ort.env.wasm.wasmBinary = gzipped
    ? await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer()
    : bytes;
  ort.env.wasm.wasmPaths = "/ort/";
  // On the GPU, one thread is enough. On the CPU, use the cores, which needs cross-origin isolation (SharedArrayBuffer).
  ort.env.wasm.numThreads =
    backend === "cpu" && self.crossOriginIsolated ? Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1)) : 1;
}

async function loadCore() {
  const bytes = await (await fetch("/wasm/driftscape.wasm")).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

async function load() {
  post({ type: "loading", detail: "Preparing the engine", fraction: 0 });
  const [, core] = await Promise.all([step("loading the ONNX runtime", loadRuntime), step("loading the core", loadCore)]);

  // Download everything first (in parallel), with one combined progress figure.
  const FILES = FILE_SETS[precision];
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

  const gpu = backend !== "cpu";
  post({ type: "loading", detail: gpu ? "Starting the model on your GPU" : "Starting the model", fraction: 1 });
  const tokenizer = new Tokenizer(JSON.parse(new TextDecoder().decode(files.tokenizer)));
  const options = { executionProviders: [gpu ? "webgpu" : "wasm"] };
  // The decoder's caches stay on the GPU between steps; only the logits come back.
  const keepOnGpu = (names) => Object.fromEntries(names.filter((n) => n.startsWith("present.")).map((n) => [n, "gpu-buffer"]));
  const decoderSession = async (file) => {
    if (!gpu) return ort.InferenceSession.create(file, options);
    const probe = await ort.InferenceSession.create(file, options);
    const outputs = probe.outputNames;
    await probe.release();
    return ort.InferenceSession.create(file, { ...options, preferredOutputLocation: keepOnGpu(outputs) });
  };
  // One at a time: onnxruntime-web refuses to create two WebGPU sessions concurrently.
  let [text, decoder, decoderPast, codec] = await step("starting the model", async () => [
    await ort.InferenceSession.create(files.text, options),
    await decoderSession(files.decoder),
    files.decoderPast ? await decoderSession(files.decoderPast) : null,
    await ort.InferenceSession.create(files.codec, options),
  ]);
  model = { tokenizer, text, decoder, decoderPast, codec, core };

  // Self-check, which also warms up the GPU (the first run compiles its programs). A browser's GPU stack can run the
  // model and still compute it wrongly: onnxruntime-web's JavaScript WebGPU implementation decodes EnCodec into
  // something that isn't music. So check against the native engine's output and route around what's broken.
  post({ type: "loading", detail: "Checking the engine", fraction: 1 });
  // 1. The audio decoder: decode the reference tokens and compare the sound with the native decode of the same tokens.
  if (!(await step("checking the audio decoder", codecMatches))) {
    if (!gpu) throw new Error("self-check failed: the audio decoder doesn't match the reference");
    post({ type: "loading", detail: "Checking the engine (audio decoding on the CPU)", fraction: 1 });
    model.codec = codec = await ort.InferenceSession.create(files.codec, { executionProviders: ["wasm"] });
    if (!(await step("checking the audio decoder", codecMatches))) {
      throw new Error("self-check failed: the audio decoder doesn't match the reference");
    }
  }
  // 2. The music decoder: greedy decoding of the reference prompt must open with the reference tokens.
  const first = await step("checking the music decoder", () =>
    generate(-1, { prompt: REFERENCE.prompt, seconds: 1, guidance: 3, temperature: 1, topK: 1, seed: 1, continueFrom: null }),
  );
  if (!skipCheck && (first.codebooks[0][0] !== REFERENCE.codebooks[0][0] || first.codebooks[0][1] !== REFERENCE.codebooks[0][1])) {
    throw new Error("self-check failed: the music decoder doesn't match the reference");
  }
  // 3. Warm up the batched replay that continuations use (other shapes, other GPU programs).
  await step("warming up", () =>
    generate(-1, { prompt: REFERENCE.prompt, seconds: 0.4, guidance: 3, temperature: 1, topK: 50, seed: 1, continueFrom: first.codebooks.map((c) => c.slice(-10)) }),
  );
}

/** Whether decoding the reference tokens sounds like the native engine's decode of them (level and zero crossings). */
async function codecMatches() {
  const frames = REFERENCE.codebooks[0].length;
  const codes = BigInt64Array.from(REFERENCE.codebooks.flat(), (v) => BigInt(v));
  const out = await model.codec.run({ audio_codes: new ort.Tensor("int64", codes, [1, 1, CODEBOOKS, frames]) });
  const { rms, zcr } = stats(out.audio_values.data);
  return Math.abs(rms - REFERENCE.rms) / REFERENCE.rms < 0.15 && Math.abs(zcr - REFERENCE.zcr) / REFERENCE.zcr < 0.25;
}

function stats(x) {
  let sum = 0;
  let crossings = 0;
  for (let i = 0; i < x.length; i++) {
    sum += x[i] * x[i];
    if (i && (x[i] < 0) !== (x[i - 1] < 0)) crossings++;
  }
  return { rms: Math.sqrt(sum / x.length), zcr: crossings / x.length };
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
  const { tokenizer, text, decoder, decoderPast, codec } = model;
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
    const ids = new ort.Tensor("int64", flat, [2 * CODEBOOKS, n]);
    let out;
    if (decoderPast) {
      // Two decoders: the first pass takes the text and every known position; later steps take the caches.
      if (!cache) {
        out = await decoder.run({ input_ids: ids, encoder_hidden_states: statesT, encoder_attention_mask: maskT });
      } else {
        const feeds = { input_ids: ids, encoder_attention_mask: maskT };
        for (const [name, value] of Object.entries(cache)) feeds[name] = value;
        out = await decoderPast.run(feeds);
      }
    } else {
      const feeds = {
        input_ids: ids,
        encoder_hidden_states: statesT,
        encoder_attention_mask: maskT,
        use_cache_branch: new ort.Tensor("bool", [cache !== null], [1]),
      };
      for (let l = 0; l < LAYERS; l++)
        for (const part of ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]) {
          const name = `past_key_values.${l}.${part}`;
          feeds[name] = cache ? cache[name] : new ort.Tensor("float32", new Float32Array(0), [2, HEADS, 0, HEAD_DIM]);
        }
      out = await decoder.run(feeds);
    }
    // Self-attention caches grow each step; cross-attention caches come from the first pass only (on cached passes
    // the merged decoder returns placeholders for them).
    const next = {};
    for (let l = 0; l < LAYERS; l++)
      for (const part of ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]) {
        const past = `past_key_values.${l}.${part}`;
        const present = out[`present.${l}.${part}`];
        if (cache && part.startsWith("encoder")) {
          next[past] = cache[past];
          present?.dispose(); // the step decoder doesn't return cross-attention caches at all
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
  // Repeats at least 8 s long where the segment allows it, so a repeat is a phrase returning, not a stutter.
  const latest = Math.max(0, from - Math.min(8 * SAMPLE_RATE, Math.max(0, from - SAMPLE_RATE)));
  const bar = analysis.bpm ? Math.round(((4 * 60) / analysis.bpm) * SAMPLE_RATE) : 0;
  // Up to three rejoin points, one per third of the allowed range, so successive repeats don't play the same loop.
  const rejoins = [];
  const earliest = Math.min(overlap, latest);
  for (let i = 0; i < 3; i++) {
    const lo = Math.round(earliest + ((latest - earliest) * i) / 3);
    const hi = Math.round(earliest + ((latest - earliest) * (i + 1)) / 3);
    let r = Number(core.rejoin(from, window, lo, hi, bar));
    if (r < 0 && bar) r = Number(core.rejoin(from, window, lo, hi, 0));
    if (r >= 0 && rejoins.every((x) => Math.abs(x - r) > SAMPLE_RATE)) rejoins.push(r);
  }
  if (!rejoins.length) {
    const r = Number(core.rejoin(from, window, 0, latest, 0));
    rejoins.push(r < 0 ? overlap : r);
  }
  const bpm = analysis.bpm ? core.tempo(SAMPLE_RATE) : 0;
  return { rejoin: rejoins[0], rejoins, bpm };
}

// ---- messages ------------------------------------------------------------------------------------------------------

function post(message, transfer = []) {
  self.postMessage(message, transfer);
}

self.onmessage = async ({ data }) => {
  if (data.type === "load") {
    backend = data.backend ?? "webgpu";
    precision = data.precision ?? "q4";
    skipCheck = !!data.skipCheck;
    try {
      await load();
      post({ type: "ready" });
    } catch (e) {
      post({ type: "error", message: e.message ?? String(e) });
    }
  } else if (data.type === "selftest") {
    // Greedy decoding of a fixed prompt, as crates/infer/examples/greedy.rs does natively: matching tokens mean this
    // backend computes the same model. Also a zero-crossing rate: about 0.03 for music, about 0.5 for noise.
    try {
      const out = await generate(-2, { prompt: "warm ambient pads, soft piano", seconds: 1, guidance: 3, temperature: 1, topK: 1, seed: 1, continueFrom: null });
      const { rms, zcr } = stats(out.audio);
      post({ type: "selftest", backend, codebooks: out.codebooks.map((c) => c.slice(0, 12)), rms, zcr });
    } catch (e) {
      post({ type: "selftest", backend, error: e.message ?? String(e) });
    }
  } else if (data.type === "bench") {
    // Silent speed test: generate `seconds` of new audio (and a continuation of it) and report seconds per second.
    try {
      const req = { prompt: "warm ambient pads, soft piano", seconds: data.seconds ?? 6, guidance: 3, temperature: 1, topK: 250, seed: 7, continueFrom: null };
      let t = performance.now();
      const a = await generate(-3, req);
      const fresh = (performance.now() - t) / 1000;
      t = performance.now();
      await generate(-3, { ...req, continueFrom: a.codebooks.map((c) => c.slice(-150)) });
      const cont = (performance.now() - t) / 1000;
      const s = stats(a.audio);
      post({ type: "bench", backend, precision, fresh: req.seconds / fresh, continued: req.seconds / cont, rms: s.rms, zcr: s.zcr });
    } catch (e) {
      post({ type: "bench", backend, precision, error: e.message ?? String(e) });
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
      const { rejoin, rejoins, bpm } = analyse(audio, overlap, request.analysis);
      post(
        {
          type: "result",
          id,
          audio,
          sampleRate: SAMPLE_RATE,
          codebooks,
          overlapSeconds,
          rejoin,
          rejoins,
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
