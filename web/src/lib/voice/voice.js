// A voice generates segments. Two implementations share one interface, so the live engine doesn't care where the
// model runs:
//   - WorkerVoice: MusicGen in this browser, on WebGPU (musicgen.worker.js). Used on the hosted app.
//   - NativeVoice: the `driftscape` engine on this machine, through `driftscape serve` (faster, runs on Metal).
//
// Request: { prompt, seconds, guidance, temperature, topK, seed, continueFrom: number[4][] | null,
//            analysis: { crossfadeSeconds, bpm | null } }
// Result:  { audio: Float32Array, sampleRate, codebooks: number[4][], overlapSeconds, rejoin (frame), bpm (0 = none),
//            seconds (generation time) }
// generate() rejects with a CancelledError when the signal aborts.

export class CancelledError extends Error {
  constructor() {
    super("cancelled");
    this.name = "CancelledError";
  }
}

/** Whether an error is a crash inside the engine (a wasm trap), as opposed to an ordinary failure. */
const isCrash = (message) => /out of bounds|unreachable|RuntimeError|memory access|Aborted|self-check failed/i.test(message);

export class WorkerVoice {
  #worker;
  #pending = new Map(); // id → { resolve, reject, onProgress }
  #next = 1;
  #ready;
  #onLoading;
  #attempts;

  /**
   * @param {(status: {detail: string, fraction: number}) => void} onLoading
   * @param {{backend: "webgpu" | "webgpu-js" | "cpu", precision: "q4" | "f16"}[]} attempts setups to try in order; a
   *   failed load, a failed self-check or a crash moves on to the next
   */
  constructor(onLoading, attempts) {
    this.#onLoading = onLoading;
    this.#attempts = [...attempts];
    this.#ready = this.#tryNext();
  }

  get label() {
    return this.backend === "cpu" ? "your browser (on the CPU)" : "your browser (on the GPU, WebGPU)";
  }

  ready() {
    return this.#ready;
  }

  /** Try the next setup, and the ones after it if it fails. */
  async #tryNext(previous) {
    const attempt = this.#attempts.shift();
    if (!attempt) throw previous;
    if (previous) {
      this.#onLoading({ detail: attempt.backend === "cpu" ? "Switching to the CPU" : "Switching to a safer setting", fraction: 0 });
    }
    try {
      await this.#start(attempt);
    } catch (e) {
      return this.#tryNext(e);
    }
  }

  /** Start a worker on a setup; resolves when the model is loaded and has passed its self-check. */
  #start({ backend, precision }) {
    this.#worker?.terminate();
    this.backend = backend;
    this.precision = precision;
    const worker = new Worker(new URL("./musicgen.worker.js", import.meta.url), { type: "module" });
    this.#worker = worker;
    return new Promise((resolve, reject) => {
      worker.onmessage = ({ data }) => {
        const job = this.#pending.get(data.id);
        switch (data.type) {
          case "loading":
            this.#onLoading({ detail: data.detail, fraction: data.fraction });
            break;
          case "ready":
            resolve();
            break;
          case "progress":
            job?.onProgress?.(data.step / data.total);
            break;
          case "result":
            this.#pending.delete(data.id);
            job?.resolve(data);
            break;
          case "cancelled":
            this.#pending.delete(data.id);
            job?.reject(new CancelledError());
            break;
          case "error":
            if (job) {
              this.#pending.delete(data.id);
              job.reject(new Error(data.message));
            } else reject(new Error(data.message));
            break;
        }
      };
      // The worker script itself failed to load or crashed.
      worker.onerror = (e) => {
        e.preventDefault?.();
        const error = new Error(`the music engine stopped${e.message ? `: ${e.message}` : ""}`);
        reject(error);
        for (const job of this.#pending.values()) job.reject(error);
        this.#pending.clear();
      };
      worker.postMessage({ type: "load", backend, precision });
    });
  }

  async generate(request, onProgress, signal) {
    try {
      return await this.#run(request, onProgress, signal);
    } catch (e) {
      // A crash mid-session (Firefox's WebGPU, for one): move on to the next setup and try again.
      if (!(e instanceof CancelledError) && this.#attempts.length && isCrash(e.message)) {
        this.#ready = this.#tryNext(e);
        await this.#ready;
        return this.#run(request, onProgress, signal);
      }
      throw e;
    }
  }

  #run(request, onProgress, signal) {
    const id = this.#next++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject, onProgress });
      signal?.addEventListener("abort", () => this.#worker.postMessage({ type: "cancel", id }), { once: true });
      this.#worker.postMessage({ type: "generate", id, request });
    });
  }
}

/** The native engine behind `driftscape serve`. */
export class NativeVoice {
  label = "the Driftscape engine on this computer";
  fast = true; // faster than real time on a recent Mac

  ready() {
    return Promise.resolve();
  }

  async generate(request, onProgress, signal) {
    onProgress?.(0);
    let res;
    try {
      res = await fetch("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(request),
        signal,
      });
    } catch (e) {
      if (signal?.aborted) throw new CancelledError();
      throw e;
    }
    if (res.status === 499 || signal?.aborted) throw new CancelledError();
    if (!res.ok) throw new Error(`the engine said: ${await res.text()}`);
    // The body is a JSON header, a newline, then the samples as little-endian f32.
    const bytes = new Uint8Array(await res.arrayBuffer());
    const newline = bytes.indexOf(10);
    const meta = JSON.parse(new TextDecoder().decode(bytes.subarray(0, newline)));
    const audio = new Float32Array(bytes.slice(newline + 1).buffer);
    onProgress?.(1);
    return { ...meta, audio };
  }
}

/**
 * The native engine if this page is served by `driftscape serve`, otherwise the in-browser one: on the GPU where the
 * browser's WebGPU works with onnxruntime-web (in Firefox through its JavaScript WebGPU implementation), on the CPU
 * elsewhere. `?engine=cpu|gpu|jsep` and `?precision=q4|f16` in the address force a setup, for testing.
 */
export async function chooseVoice(onLoading) {
  const forced = new URLSearchParams(location.search).get("engine");
  if (!forced) try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (res.ok && (await res.json()).engine === "driftscape") return new NativeVoice();
  } catch {
    // no native engine here
  }
  const params = new URLSearchParams(location.search);
  const adapter = "gpu" in navigator ? await navigator.gpu.requestAdapter().catch(() => null) : null;
  // Firefox: onnxruntime-web's native WebGPU implementation crashes there, so try its JavaScript one. Everywhere else:
  // native WebGPU when there is an adapter. Then the CPU.
  let backend = { cpu: "cpu", gpu: "webgpu", jsep: "webgpu-js" }[forced] ?? null;
  if (!backend) backend = !adapter ? "cpu" : /firefox/i.test(navigator.userAgent) ? "webgpu-js" : "webgpu";
  // Float16 arithmetic is opt-in (`?precision=f16`, diagnostics): the fp16 export fails the self-check (MusicGen
  // overflows in float16), so offering it by default would only cost a 635 MB download before falling back.
  const f16 = backend !== "cpu" && params.get("precision") === "f16";
  const attempts = [
    ...(f16 ? [{ backend, precision: "f16" }] : []),
    ...(backend !== "cpu" ? [{ backend, precision: "q4" }] : []),
    { backend: "cpu", precision: "q4" },
  ];
  return new WorkerVoice(onLoading, forced ? attempts.slice(0, 1) : attempts);
}

/** A message for the listener when an engine error is a crash rather than an ordinary failure. */
export function explain(message) {
  return isCrash(message)
    ? "The music engine crashed in this browser. Reload to try again; if it keeps happening, Chrome, Edge or Safari, or `driftscape serve`, will run it."
    : message;
}
