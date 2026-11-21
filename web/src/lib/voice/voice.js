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

export class WorkerVoice {
  label = "your browser (WebGPU)";
  #worker;
  #pending = new Map(); // id → { resolve, reject, onProgress }
  #next = 1;
  #ready;

  /** @param {(status: {detail: string, fraction: number}) => void} onLoading */
  constructor(onLoading) {
    this.#worker = new Worker(new URL("./musicgen.worker.js", import.meta.url), { type: "module" });
    let readyResolve, readyReject;
    this.#ready = new Promise((res, rej) => ((readyResolve = res), (readyReject = rej)));
    this.#worker.onmessage = ({ data }) => {
      const job = this.#pending.get(data.id);
      switch (data.type) {
        case "loading":
          onLoading({ detail: data.detail, fraction: data.fraction });
          break;
        case "ready":
          readyResolve();
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
          } else readyReject(new Error(data.message));
          break;
      }
    };
    // The worker script itself failed to load or crashed.
    this.#worker.onerror = (e) => {
      e.preventDefault?.();
      const error = new Error(`the music engine failed to start${e.message ? `: ${e.message}` : ""}`);
      readyReject(error);
      for (const job of this.#pending.values()) job.reject(error);
      this.#pending.clear();
    };
    this.#worker.postMessage({ type: "load" });
  }

  ready() {
    return this.#ready;
  }

  generate(request, onProgress, signal) {
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

/** The native engine if this page is served by `driftscape serve`, otherwise the in-browser one. */
export async function chooseVoice(onLoading) {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (res.ok && (await res.json()).engine === "driftscape") return new NativeVoice();
  } catch {
    // no native engine here
  }
  return new WorkerVoice(onLoading);
}
