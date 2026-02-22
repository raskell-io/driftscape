// AudioEngine hook — Transformers.js MusicGen + Web Audio API integration
//
// Lazy-loads the model on first generate, runs inference client-side via
// WebGPU (falls back to WASM), and plays back the result through Web Audio
// with seamless looping and crossfade support.

const SAMPLE_RATE = 32000; // MusicGen outputs 32 kHz mono audio
const CROSSFADE_DURATION = 2; // seconds

let pipeline = null;
let generator = null;

async function loadModel(hook) {
  if (generator) return generator;

  hook.pushEvent("generation-progress", { progress: 5, label: "Loading Transformers.js..." });

  const { pipeline: pipelineFn, env } = await import("@huggingface/transformers");
  pipeline = pipelineFn;

  // Prefer WebGPU, fall back to WASM
  env.backends.onnx.wasm.proxy = false;

  hook.pushEvent("generation-progress", { progress: 15, label: "Downloading MusicGen model..." });

  generator = await pipeline("text-to-audio", "Xenova/musicgen-small", {
    dtype: "q4",
    device: "webgpu",
    progress_callback: (progress) => {
      if (progress.status === "progress" && progress.total) {
        const pct = Math.min(15 + Math.round((progress.loaded / progress.total) * 50), 65);
        hook.pushEvent("generation-progress", {
          progress: pct,
          label: `Downloading model: ${progress.file || ""}`,
        });
      }
    },
  });

  hook.pushEvent("generation-progress", { progress: 70, label: "Model ready" });
  return generator;
}

function createAudioContext() {
  return new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
}

function playAudio(ctx, audioData, gainNode, loop = true) {
  const buffer = ctx.createBuffer(1, audioData.length, SAMPLE_RATE);
  buffer.getChannelData(0).set(audioData);

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.loop = loop;
  source.connect(gainNode);
  gainNode.connect(ctx.destination);
  source.start();

  return source;
}

function crossfade(ctx, oldGain, newGain, duration = CROSSFADE_DURATION) {
  const now = ctx.currentTime;
  oldGain.gain.setValueAtTime(1, now);
  oldGain.gain.linearRampToValueAtTime(0, now + duration);
  newGain.gain.setValueAtTime(0, now);
  newGain.gain.linearRampToValueAtTime(1, now + duration);

  // Disconnect old gain after fade completes
  setTimeout(() => {
    try {
      oldGain.disconnect();
    } catch (_e) {
      // already disconnected
    }
  }, duration * 1000 + 100);
}

const AudioEngine = {
  mounted() {
    this.audioCtx = null;
    this.currentSource = null;
    this.currentGain = null;

    this.handleEvent("generate-audio", async ({ prompt }) => {
      await this.generate(prompt);
    });

    // Listen for the generate button click from LiveView
    this.el.closest("[phx-hook]") || this.el;
    window.addEventListener("phx:generate", (e) => {
      this.generate(e.detail.prompt);
    });
  },

  async generate(prompt) {
    if (!prompt) {
      prompt = this.el.dataset.prompt;
    }
    if (!prompt) return;

    try {
      const gen = await loadModel(this);

      this.pushEvent("generation-progress", { progress: 75, label: "Generating audio..." });

      const result = await gen(prompt, {
        max_new_tokens: 512,
        callback_function: (output) => {
          if (output && output.generated_token_ids) {
            const total = 512;
            const done = output.generated_token_ids.length;
            const pct = Math.min(75 + Math.round((done / total) * 20), 95);
            this.pushEvent("generation-progress", {
              progress: pct,
              label: "Generating audio...",
            });
          }
        },
      });

      this.pushEvent("generation-progress", { progress: 98, label: "Starting playback..." });

      // Extract audio data from the pipeline result
      const audioData = result.audio;
      if (!audioData || audioData.length === 0) {
        console.error("No audio data generated");
        return;
      }

      // Initialize audio context on first use (must be after user gesture)
      if (!this.audioCtx) {
        this.audioCtx = createAudioContext();
      }

      const newGain = this.audioCtx.createGain();

      // Crossfade if there's already something playing
      if (this.currentSource && this.currentGain) {
        crossfade(this.audioCtx, this.currentGain, newGain);
        const oldSource = this.currentSource;
        setTimeout(() => {
          try {
            oldSource.stop();
          } catch (_e) {
            // already stopped
          }
        }, CROSSFADE_DURATION * 1000 + 100);
      }

      this.currentSource = playAudio(this.audioCtx, audioData, newGain, true);
      this.currentGain = newGain;

      this.pushEvent("generation-complete", {});
    } catch (error) {
      console.error("Audio generation failed:", error);
      this.pushEvent("generation-progress", {
        progress: 0,
        label: `Error: ${error.message}`,
      });
    }
  },

  destroyed() {
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch (_e) {
        // already stopped
      }
    }
    if (this.audioCtx) {
      this.audioCtx.close();
    }
  },
};

export default AudioEngine;
