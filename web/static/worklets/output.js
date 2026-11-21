// The output stage, on the audio thread: a slow leveler with a soft limiter (a port of driftscape_core::mixer::Leveler;
// see that file for why), and a tap that hands the leveled samples to the page while it records.

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
    const input = inputs[0];
    const output = outputs[0];
    if (!input.length) return true;
    const mono = input[0];
    const out = output[0];
    const floor = TARGET_RMS / (2 * MAX_GAIN);
    for (let i = 0; i < mono.length; i++) {
      const s = mono[i];
      this.power = this.coef * this.power + (1 - this.coef) * s * s;
      const rms = Math.max(Math.sqrt(this.power), floor);
      const wanted = Math.min(Math.max(TARGET_RMS / rms, MIN_GAIN), MAX_GAIN);
      this.gain = this.coef * this.gain + (1 - this.coef) * wanted;
      out[i] = softLimit(s * this.gain);
    }
    for (let c = 1; c < output.length; c++) output[c].set(out);
    if (this.recording) this.port.postMessage(out.slice());
    return true;
  }
}

registerProcessor("driftscape-output", Output);
