// AudioEngine hook — Transformers.js MusicGen + Web Audio API integration
//
// The model downloads in the background on page load. The user focuses on
// writing their prompt. A toast reports model readiness. Generation is only
// enabled once the model is ready.

const CROSSFADE_DURATION = 2; // seconds

let modelPromise = null;

function loadModel(hook) {
  if (modelPromise) return modelPromise;

  modelPromise = (async () => {
    hook.pushEvent("model-status", { status: "loading", detail: "Loading Transformers.js..." });

    const { AutoTokenizer, MusicgenForConditionalGeneration } = await import("@huggingface/transformers");

    hook.pushEvent("model-status", { status: "loading", detail: "Downloading tokenizer..." });

    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/musicgen-small");

    hook.pushEvent("model-status", { status: "downloading", detail: "Downloading MusicGen model..." });

    const model = await MusicgenForConditionalGeneration.from_pretrained("Xenova/musicgen-small", {
      dtype: "q4",
      device: "webgpu",
      progress_callback: (progress) => {
        if (progress.status === "progress" && progress.total) {
          const pct = Math.round((progress.loaded / progress.total) * 100);
          hook.pushEvent("model-status", {
            status: "downloading",
            detail: `Downloading model: ${pct}%`,
          });
        }
      },
    });

    const sampleRate = model.config.audio_encoder.sampling_rate;

    hook.pushEvent("model-status", { status: "ready", detail: "" });

    return { tokenizer, model, sampleRate };
  })();

  modelPromise.catch(() => { modelPromise = null; });

  return modelPromise;
}

function crossfade(ctx, oldGain, newGain, duration = CROSSFADE_DURATION) {
  const now = ctx.currentTime;
  oldGain.gain.setValueAtTime(1, now);
  oldGain.gain.linearRampToValueAtTime(0, now + duration);
  newGain.gain.setValueAtTime(0, now);
  newGain.gain.linearRampToValueAtTime(1, now + duration);

  setTimeout(() => {
    try { oldGain.disconnect(); } catch (_e) { /* already disconnected */ }
  }, duration * 1000 + 100);
}

const AudioEngine = {
  mounted() {
    this.audioCtx = null;
    this.currentSource = null;
    this.currentGain = null;

    // Start downloading immediately in the background
    loadModel(this);

    this.handleEvent("generate-audio", ({ prompt }) => {
      this.generate(prompt);
    });

    this.handleEvent("stop-audio", () => {
      if (this.currentSource) {
        try { this.currentSource.stop(); } catch (_e) { /* already stopped */ }
        this.currentSource = null;
      }
      if (this.currentGain) {
        try { this.currentGain.disconnect(); } catch (_e) { /* already disconnected */ }
        this.currentGain = null;
      }
    });
  },

  async generate(prompt) {
    if (!prompt) return;

    try {
      const { tokenizer, model, sampleRate } = await loadModel(this);

      this.pushEvent("generation-progress", { progress: 50, label: "Generating audio..." });

      const inputs = tokenizer(prompt);
      const audioValues = await model.generate({
        ...inputs,
        max_new_tokens: 512,
        do_sample: true,
        guidance_scale: 3,
      });

      this.pushEvent("generation-progress", { progress: 95, label: "Starting playback..." });

      const audioData = audioValues.data;
      if (!audioData || audioData.length === 0) {
        this.pushEvent("generation-progress", { progress: 0, label: "Error: no audio generated" });
        return;
      }

      if (!this.audioCtx) {
        this.audioCtx = new AudioContext({ sampleRate });
      }

      const buffer = this.audioCtx.createBuffer(1, audioData.length, sampleRate);
      buffer.getChannelData(0).set(audioData instanceof Float32Array ? audioData : new Float32Array(audioData));

      const newGain = this.audioCtx.createGain();
      const source = this.audioCtx.createBufferSource();
      source.buffer = buffer;
      source.loop = true;
      source.connect(newGain);
      newGain.connect(this.audioCtx.destination);

      if (this.currentSource && this.currentGain) {
        crossfade(this.audioCtx, this.currentGain, newGain);
        const oldSource = this.currentSource;
        setTimeout(() => {
          try { oldSource.stop(); } catch (_e) { /* already stopped */ }
        }, CROSSFADE_DURATION * 1000 + 100);
      }

      source.start();
      this.currentSource = source;
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
      try { this.currentSource.stop(); } catch (_e) { /* already stopped */ }
    }
    if (this.audioCtx) {
      this.audioCtx.close();
    }
  },
};

export default AudioEngine;
