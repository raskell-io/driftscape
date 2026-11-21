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
  #fellBack = false;

  /**
   * @param {(status: {detail: string, fraction: number}) => void} onLoading
   * @param {"webgpu" | "webgpu-js" | "cpu"} backend
   */
  constructor(onLoading, backend) {
    this.#onLoading = onLoading;
    this.backend = backend;
    this.#ready = this.#start(backend).catch((e) => this.#fallBack(e));
  }

  get label() {
    return this.backend === "cpu" ? "your browser (on the CPU)" : "your browser (on the GPU, WebGPU)";
  }

  ready() {
    return this.#ready;
  }

  /** Start a worker on a backend; resolves when the model is loaded. */
  #start(backend) {
    this.#worker?.terminate();
    this.backend = backend;
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
      worker.postMessage({ type: "load", backend });
    });
  }

  /** WebGPU didn't work here: carry on on the CPU, once. */
  async #fallBack(error) {
    if (this.backend === "cpu" || this.#fellBack) throw error;
    this.#fellBack = true;
    this.#onLoading({ detail: "Your browser's GPU didn't work with the engine; switching to the CPU", fraction: 0 });
    await this.#start("cpu");
  }

  async generate(request, onProgress, signal) {
    try {
      return await this.#run(request, onProgress, signal);
    } catch (e) {
      // A crash on the GPU path (Firefox's WebGPU, for one): switch to the CPU and try again.
      if (!(e instanceof CancelledError) && this.backend !== "cpu" && !this.#fellBack && isCrash(e.message)) {
        this.#ready = this.#fallBack(e);
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
 * elsewhere. `?engine=cpu`, `?engine=gpu` or `?engine=jsep` in the address overrides the choice, for testing.
 */
export async function chooseVoice(onLoading) {
  const forced = new URLSearchParams(location.search).get("engine");
  if (!forced) try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (res.ok && (await res.json()).engine === "driftscape") return new NativeVoice();
  } catch {
    // no native engine here
  }
  let backend = { cpu: "cpu", gpu: "webgpu", jsep: "webgpu-js" }[forced] ?? null;
  if (!backend) {
    // Firefox: onnxruntime-web's native WebGPU implementation crashes there, so try its JavaScript one (falling back to
    // the CPU if that crashes too). Everywhere else: native WebGPU when there is an adapter, else the CPU.
    const firefox = /firefox/i.test(navigator.userAgent);
    const adapter = "gpu" in navigator ? await navigator.gpu.requestAdapter().catch(() => null) : null;
    backend = !adapter ? "cpu" : firefox ? "webgpu-js" : "webgpu";
  }
  return new WorkerVoice(onLoading, backend);
}

/** A message for the listener when an engine error is a crash rather than an ordinary failure. */
export function explain(message) {
  return isCrash(message)
    ? "The music engine crashed in this browser. Reload to try again; if it keeps happening, Chrome, Edge or Safari, or `driftscape serve`, will run it."
    : message;
}
