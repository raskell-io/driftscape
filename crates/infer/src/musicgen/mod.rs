//! MusicGen (small).
//!
//! The pipeline, as in Hugging Face `transformers`:
//!
//! 1. A T5 encoder turns the prompt into hidden states (ONNX Runtime; it takes milliseconds).
//! 2. A decoder predicts EnCodec audio tokens, 50 per second, in 4 codebooks. The codebooks are staggered by one step
//!    each (the "delay pattern"): at step `p`, codebook `k` predicts frame `p - k - 1`. Positions outside a
//!    codebook's window are forced to the pad token. This is nearly all of the work, so it has two backends behind
//!    [`Decoder`]: ONNX Runtime on the CPU ([`onnx`]) and candle on the GPU ([`gpu`]).
//! 3. Classifier-free guidance runs a second, unconditional batch alongside (zeroed text states) and pushes the
//!    logits away from it: `uncond + guidance * (cond - uncond)`.
//! 4. The EnCodec decoder turns the 4 × frames tokens into 32 kHz mono audio (ONNX Runtime).

pub mod gpu;
pub mod onnx;

use crate::{
    Cancelled, Clip, ControlFlow, Progress, Request, Tokens, Voice,
    sampling::{Rng, top_k},
};
use anyhow::{Context, Result, anyhow, bail};
use driftscape_core::Audio;
use ort::{environment::Environment, session::Session, value::Tensor};
use std::path::Path;
use tokenizers::Tokenizer;

const CODEBOOKS: usize = 4;
const VOCAB: usize = 2048;
const PAD: i64 = 2048;
const TEXT_DIM: usize = 768;
const FRAME_RATE: f32 = 50.0;
const SAMPLE_RATE: u32 = 32_000;
/// MusicGen was trained on 30-second clips; it degrades beyond that.
pub const MAX_SECONDS: f32 = 30.0;

/// The token decoder: given the text states once, then the previous tokens each step, return logits.
pub trait Decoder {
    /// Start a new sequence. `states` is `[2, seq, 768]` (conditional, then unconditional batch), `mask` is `[2, seq]`.
    fn start(&mut self, states: &[f32], mask: &[i64], seq: usize) -> Result<()>;
    /// Feed one token per row (`2 * CODEBOOKS` rows: batch-major, codebook-minor) and return the logits for the next
    /// position, `[2 * CODEBOOKS, VOCAB]` in the same row order.
    fn step(&mut self, tokens: &[i64]) -> Result<Vec<f32>>;
    /// Feed several consecutive positions whose tokens are all known (a continuation's replayed overlap) and return
    /// the logits after the last. Backends that can run them as one batch should; this default steps through them.
    fn prefill(&mut self, positions: &[Vec<i64>]) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        for tokens in positions {
            logits = self.step(tokens)?;
        }
        Ok(logits)
    }
}

/// Which decoder backend to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// ONNX Runtime on the CPU, using the q4 ONNX files.
    Cpu,
    /// candle on the GPU (Metal on macOS), using the safetensors weights.
    Gpu,
}

pub struct MusicGen {
    tokenizer: Tokenizer,
    text_encoder: Session,
    codec: Session,
    decoder: Box<dyn Decoder>,
}

fn trace(what: &str, since: std::time::Instant) {
    if std::env::var_os("DRIFTSCAPE_TRACE").is_some() {
        eprintln!("[trace] {what} {:.2} s", since.elapsed().as_secs_f32());
    }
}

pub(crate) fn session(env: &Environment, path: &Path) -> Result<Session> {
    let mut builder = Session::builder(env).map_err(|e| anyhow!("{e}"))?;
    if let Some(n) = std::env::var("DRIFTSCAPE_THREADS").ok().and_then(|v| v.parse().ok()) {
        builder = builder.with_intra_threads(n).map_err(|e| anyhow!("{e}"))?;
    }
    builder.commit_from_file(path).with_context(|| format!("loading {}", path.display()))
}

impl MusicGen {
    /// Load from a model directory laid out like the Hugging Face repo `Xenova/musicgen-small`
    /// (`tokenizer.json`, `onnx/*_q4.onnx`), plus `safetensors/model.safetensors` from `facebook/musicgen-small` for
    /// the GPU backend.
    pub fn load(env: &Environment, dir: &Path, backend: Backend) -> Result<Self> {
        let started = std::time::Instant::now();
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer.json from {}: {e}", dir.display()))?;
        let onnx = dir.join("onnx");
        let decoder: Box<dyn Decoder> = match backend {
            Backend::Cpu => Box::new(onnx::OnnxDecoder::load(env, &onnx.join("decoder_model_merged_q4.onnx"))?),
            Backend::Gpu => {
                let d = gpu::GpuDecoder::load(&dir.join("safetensors/model.safetensors"))?;
                if !d.on_gpu() {
                    eprintln!("warning: no GPU available; the decoder runs on the CPU through candle");
                }
                Box::new(d)
            }
        };
        trace("decoder load", started);
        Ok(Self {
            tokenizer,
            text_encoder: session(env, &onnx.join("text_encoder_q4.onnx"))?,
            codec: session(env, &onnx.join("encodec_decode_q4.onnx"))?,
            decoder,
        })
    }

    /// Text → hidden states `[1, seq, 768]` and the sequence length.
    fn encode_text(&mut self, prompt: &str) -> Result<(Vec<f32>, usize)> {
        let encoding = self.tokenizer.encode(prompt, true).map_err(|e| anyhow!("tokenizing the prompt: {e}"))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i64::from(i)).collect();
        let len = ids.len();
        let outputs = self.text_encoder.run(ort::inputs![
            "input_ids" => Tensor::from_array(([1, len], ids))?,
            "attention_mask" => Tensor::from_array(([1, len], vec![1i64; len]))?,
        ])?;
        let (_, hidden) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
        Ok((hidden.to_vec(), len))
    }
}

/// Whether codebook `k` must hold the pad token at sequence position `p` of a sequence of `len` positions.
fn forced_pad(k: usize, p: usize, len: usize) -> bool {
    p <= k || p >= len - CODEBOOKS + 1 + k
}

/// The token codebook `k` must take at position `p`, if it is not the model's to choose: the pad token outside the
/// codebook's window, or the continuation prefix's token for frames inside the prefix.
fn forced_token(k: usize, p: usize, len: usize, prefix: Option<&Tokens>) -> Option<i64> {
    if forced_pad(k, p, len) {
        return Some(PAD);
    }
    let frame = p - k - 1; // codebook k is delayed by k positions after the start token
    prefix.and_then(|t| t.codebooks[k].get(frame).copied())
}

impl Voice for MusicGen {
    fn generate(&mut self, request: &Request, progress: &mut dyn FnMut(Progress) -> ControlFlow<()>) -> Result<Clip> {
        let prefix = request.continue_from.as_ref();
        let prefix_frames = prefix.map_or(0, Tokens::frames);
        if let Some(t) = prefix
            && (t.codebooks.len() != CODEBOOKS || t.codebooks.iter().any(|c| c.len() != prefix_frames))
        {
            bail!("continuation tokens must have {CODEBOOKS} codebooks of equal length");
        }
        let new_frames = (request.seconds * FRAME_RATE).round() as usize;
        let frames = prefix_frames + new_frames;
        if new_frames == 0 || frames as f32 > MAX_SECONDS * FRAME_RATE {
            bail!(
                "MusicGen makes up to {MAX_SECONDS} seconds including the continuation; asked for {} + {:.1}",
                request.seconds,
                prefix_frames as f32 / FRAME_RATE
            );
        }
        // Sequence positions: one start token, then `frames + CODEBOOKS - 1` steps to fill the staggered codebooks.
        let len = frames + CODEBOOKS;
        let steps = len - 1;

        let t0 = std::time::Instant::now();
        let (hidden, seq) = self.encode_text(&request.prompt)?;
        trace("text encoder", t0);

        // Conditional and unconditional batches side by side.
        let mut states = hidden;
        states.extend(std::iter::repeat_n(0.0, seq * TEXT_DIM));
        let mut mask = vec![1i64; seq];
        mask.extend(std::iter::repeat_n(0, seq));
        self.decoder.start(&states, &mask, seq)?;

        let mut tokens = vec![vec![PAD; len]; CODEBOOKS]; // the conditional batch's sequence per codebook
        let mut rng = Rng::new(request.seed);
        let mut scratch = Vec::with_capacity(VOCAB);
        let mut guided = vec![0f32; VOCAB];
        // The decoder input for a position: its tokens for both batches (the same tokens, batch-major).
        let input = |tokens: &[Vec<i64>], q: usize| -> Vec<i64> {
            (0..2).flat_map(|_| tokens.iter().map(move |c| c[q])).collect()
        };

        let t0 = std::time::Instant::now();
        // Positions whose tokens are all forced (the start token and a continuation's overlap) are fed in one batch.
        let known = (0..len)
            .take_while(|&q| q == 0 || (0..CODEBOOKS).all(|k| forced_token(k, q, len, prefix).is_some()))
            .count();
        for q in 1..known {
            for (k, sequence) in tokens.iter_mut().enumerate() {
                sequence[q] = forced_token(k, q, len, prefix).expect("known positions are forced");
            }
        }
        let positions: Vec<Vec<i64>> = (0..known).map(|q| input(&tokens, q)).collect();
        let mut logits = self.decoder.prefill(&positions).context("prefill")?;
        for p in known..=steps {
            if p > known {
                logits = self.decoder.step(&input(&tokens, p - 1)).with_context(|| format!("decoder step {p}"))?;
            }
            for (k, sequence) in tokens.iter_mut().enumerate() {
                sequence[p] = if let Some(t) = forced_token(k, p, len, prefix) {
                    t
                } else {
                    let (cond, uncond) = (&logits[k * VOCAB..][..VOCAB], &logits[(CODEBOOKS + k) * VOCAB..][..VOCAB]);
                    for ((g, c), u) in guided.iter_mut().zip(cond).zip(uncond) {
                        *g = u + request.guidance * (c - u);
                    }
                    top_k(&guided, request.top_k, request.temperature, &mut rng, &mut scratch) as i64
                };
            }
            if progress(Progress { step: p, total: steps }).is_break() {
                return Err(Cancelled.into());
            }
        }
        trace("decoder", t0);

        // Undo the stagger: codebook k's frames sit at positions k+1 ..= k+frames.
        let codebooks: Vec<Vec<i64>> = (0..CODEBOOKS).map(|k| tokens[k][k + 1..=k + frames].to_vec()).collect();
        let codes: Vec<i64> = codebooks.iter().flatten().copied().collect();
        let t0 = std::time::Instant::now();
        let outputs =
            self.codec.run(ort::inputs!["audio_codes" => Tensor::from_array(([1, 1, CODEBOOKS, frames], codes))?])?;
        let (_, samples) = outputs["audio_values"].try_extract_tensor::<f32>()?;
        trace("audio codec", t0);
        Ok(Clip {
            audio: Audio::mono(SAMPLE_RATE, samples.to_vec()),
            tokens: Tokens { codebooks, frame_rate: FRAME_RATE },
            overlap_seconds: prefix_frames as f32 / FRAME_RATE,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_pattern_leaves_exactly_frames_per_codebook() {
        let frames = 10;
        let len = frames + CODEBOOKS;
        for k in 0..CODEBOOKS {
            let free: Vec<usize> = (0..len).filter(|&p| !forced_pad(k, p, len)).collect();
            assert_eq!(free, (k + 1..=k + frames).collect::<Vec<_>>(), "codebook {k}");
        }
    }

    #[test]
    fn continuation_prefix_is_replayed_in_delay_order() {
        // 3 prefix frames, 2 new: codebook k must replay prefix frame f at position f + k + 1, then be free.
        let prefix = Tokens {
            codebooks: (0..CODEBOOKS).map(|k| (0..3).map(|f| (k * 10 + f) as i64).collect()).collect(),
            frame_rate: 50.0,
        };
        let len = 5 + CODEBOOKS;
        for k in 0..CODEBOOKS {
            for f in 0..3 {
                assert_eq!(forced_token(k, f + k + 1, len, Some(&prefix)), Some((k * 10 + f) as i64));
            }
            assert_eq!(forced_token(k, 3 + k + 1, len, Some(&prefix)), None, "first new frame of codebook {k}");
            assert_eq!(forced_token(k, k, len, Some(&prefix)), Some(PAD));
        }
    }
}
