// The output stage, on the audio thread: a slow leveler for the music (a port of driftscape_core::mixer::Leveler; see
// that file for why), the ambience bed added around it in stereo (input 1, not leveled: its level is chosen), a soft
// limiter, and a tap that hands the interleaved stereo output to the page while it records.

const TARGET_RMS = 0.1;
const SECONDS = 3;
const MIN_GAIN = 0.5;
const MAX_GAIN = 4;
const KNEE = 0.8;

function softLimit(x) {
  const a = Math.abs(x);
  if (a <= KNEE) return x;
  return Math.sign(x) * (KNEE + (1 - KNEE) * Math.tanh((a - KNEE) / (1 - KNEE)));
}

class Output extends AudioWorkletProcessor {
  constructor() {
    super();
    this.coef = Math.exp(-1 / (SECONDS * sampleRate));
    this.power = TARGET_RMS * TARGET_RMS;
    this.gain = 1;
    this.recording = false;
    this.port.onmessage = ({ data }) => {
      if (data.type === "record") this.recording = data.on;
    };
  }

  process(inputs, outputs) {
    const mono = inputs[0]?.[0];
    const bed = inputs[1] ?? [];
    const [left, right = left] = outputs[0];
    const bedL = bed[0];
    const bedR = bed[1] ?? bed[0];
    const floor = TARGET_RMS / (2 * MAX_GAIN);
    for (let i = 0; i < left.length; i++) {
      let s = 0;
      if (mono) {
        s = mono[i];
        this.power = this.coef * this.power + (1 - this.coef) * s * s;
        const rms = Math.max(Math.sqrt(this.power), floor);
        const wanted = Math.min(Math.max(TARGET_RMS / rms, MIN_GAIN), MAX_GAIN);
        this.gain = this.coef * this.gain + (1 - this.coef) * wanted;
        s *= this.gain;
      }
      left[i] = softLimit(s + (bedL ? bedL[i] : 0));
      if (right !== left) right[i] = softLimit(s + (bedR ? bedR[i] : 0));
    }
    if (this.recording) {
      const frames = new Float32Array(left.length * 2);
      for (let i = 0; i < left.length; i++) {
        frames[2 * i] = left[i];
        frames[2 * i + 1] = right[i];
      }
      this.port.postMessage(frames, [frames.buffer]);
    }
    return true;
  }
}

registerProcessor("driftscape-output", Output);
