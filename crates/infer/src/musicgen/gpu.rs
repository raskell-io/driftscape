//! The decoder on candle, on the GPU (Metal) when built with the `metal` feature, with weights from
//! `facebook/musicgen-small` (`model.safetensors`).
//!
//! A pre-norm transformer decoder, written to match Hugging Face's `MusicgenForCausalLM`:
//! - the four codebooks' token embeddings are summed, plus sinusoidal positions (cos half first, then sin);
//! - 24 layers of self-attention, cross-attention to the text and a GELU feed-forward, each with a LayerNorm before it
//!   and a residual after; attention projections have no bias;
//! - one linear head per codebook.
//!
//! Text states go through `enc_to_dec_proj` (768 → 1024) and are zeroed where the attention mask is zero, as in
//! `MusicgenForConditionalGeneration`. The unconditional batch therefore attends to zeros, and because the attention
//! projections have no bias its keys and values are zero too: the output is zero with or without a mask, so no mask is
//! needed.
//!
//! Speed: one decoding step is small matrix-vector work, so the cost is the number of GPU dispatches, not arithmetic.
//! Hence the fused q/k/v projection, the fused attention kernel (`sdpa`) on Metal, the preallocated KV cache and one
//! matrix multiply for all four output heads.

use super::{CODEBOOKS, Decoder, TEXT_DIM, VOCAB};
use anyhow::{Context, Result, anyhow};
use candle_core::{
    DType, Device, Module, Tensor,
    quantized::{GgmlDType, QMatMul, QTensor},
};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, kv_cache::KvCache, layer_norm, linear};
use std::path::Path;

const HIDDEN: usize = 1024;
const LAYERS: usize = 24;
const HEADS: usize = 16;
const HEAD_DIM: usize = HIDDEN / HEADS;
const FFN: usize = 4096;
const MAX_POSITIONS: usize = 2048;
const EPS: f64 = 1e-5;

/// `[batch, seq, hidden]` → `[batch, heads, seq, head_dim]`.
fn heads(x: &Tensor) -> Result<Tensor> {
    let (b, s, _) = x.dims3()?;
    Ok(x.reshape((b, s, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?)
}

/// `softmax(q kᵀ / √d) v`, fused into one kernel on Metal.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let scale = (HEAD_DIM as f32).powf(-0.5);
    if q.device().is_metal() {
        return Ok(candle_nn::ops::sdpa(q, &k.contiguous()?, &v.contiguous()?, None, false, scale, 1.0)?);
    }
    let scores = (q.matmul(&k.t()?)? * f64::from(scale))?;
    Ok(candle_nn::ops::softmax_last_dim(&scores)?.matmul(v)?)
}

/// `[batch, heads, seq, head_dim]` → `[batch, seq, hidden]`.
fn merge(x: &Tensor) -> Result<Tensor> {
    let (b, _, s, _) = x.dims4()?;
    Ok(x.transpose(1, 2)?.reshape((b, s, HIDDEN))?)
}

/// Where weights come from: the same checkpoint on the device (for plain layers) and on the CPU in f32 (the source for
/// quantization, which candle only does on the CPU).
#[derive(Clone)]
struct Weights<'a> {
    device_vb: VarBuilder<'a>,
    cpu_vb: VarBuilder<'a>,
    device: Device,
    quant: Option<GgmlDType>,
}

impl Weights<'_> {
    fn pp(&self, prefix: impl ToString) -> Self {
        let prefix = prefix.to_string();
        Self {
            device_vb: self.device_vb.pp(&prefix),
            cpu_vb: self.cpu_vb.pp(&prefix),
            device: self.device.clone(),
            quant: self.quant,
        }
    }

    /// One or several bias-free projections of the same input, as one matrix multiply.
    fn proj(&self, names: &[&str], in_dim: usize, out_dim: usize) -> Result<Proj> {
        let shape = (out_dim, in_dim);
        let get = |vb: &VarBuilder| -> candle_core::Result<Tensor> {
            let parts =
                names.iter().map(|n| vb.get(shape, &format!("{n}.weight"))).collect::<candle_core::Result<Vec<_>>>()?;
            Tensor::cat(&parts, 0)
        };
        Ok(match self.quant {
            Some(q) => {
                Proj::Quantized(QMatMul::from_qtensor(QTensor::quantize_onto(&get(&self.cpu_vb)?, q, &self.device)?)?)
            }
            None => Proj::Plain(Linear::new(get(&self.device_vb)?, None)),
        })
    }
}

/// A bias-free linear layer, plain or with quantized weights (llama.cpp-style kernels, fast for few rows).
enum Proj {
    Plain(Linear),
    Quantized(QMatMul),
}

impl Module for Proj {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Plain(l) => l.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }
}

struct Layer {
    self_norm: LayerNorm,
    self_qkv: Proj,
    self_out: Proj,
    cross_norm: LayerNorm,
    cross_q: Proj,
    cross_kv: Proj,
    cross_out: Proj,
    ffn_norm: LayerNorm,
    fc1: Proj,
    fc2: Proj,
    cache: KvCache,
    /// Cross-attention keys and values, computed once per sequence from the text.
    cross: Option<(Tensor, Tensor)>,
}

impl Layer {
    fn load(w: Weights, max_len: usize) -> Result<Self> {
        let (sa, ca) = (w.pp("self_attn"), w.pp("encoder_attn"));
        let vb = &w.device_vb;
        Ok(Self {
            self_norm: layer_norm(HIDDEN, EPS, vb.pp("self_attn_layer_norm"))?,
            self_qkv: sa.proj(&["q_proj", "k_proj", "v_proj"], HIDDEN, HIDDEN)?,
            self_out: sa.proj(&["out_proj"], HIDDEN, HIDDEN)?,
            cross_norm: layer_norm(HIDDEN, EPS, vb.pp("encoder_attn_layer_norm"))?,
            cross_q: ca.proj(&["q_proj"], HIDDEN, HIDDEN)?,
            cross_kv: ca.proj(&["k_proj", "v_proj"], HIDDEN, HIDDEN)?,
            cross_out: ca.proj(&["out_proj"], HIDDEN, HIDDEN)?,
            ffn_norm: layer_norm(HIDDEN, EPS, vb.pp("final_layer_norm"))?,
            fc1: w.proj(&["fc1"], HIDDEN, FFN)?,
            fc2: w.proj(&["fc2"], FFN, HIDDEN)?,
            cache: KvCache::new(2, max_len),
            cross: None,
        })
    }

    fn start(&mut self, states: &Tensor, max_len: usize) -> Result<()> {
        let kv = self.cross_kv.forward(states)?;
        self.cross = Some((heads(&kv.narrow(2, 0, HIDDEN)?)?, heads(&kv.narrow(2, HIDDEN, HIDDEN)?)?));
        self.cache = KvCache::new(2, max_len);
        Ok(())
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        // Self-attention over everything so far. One new position per step, so no causal mask is needed.
        let h = self.self_norm.forward(x)?;
        let qkv = self.self_qkv.forward(&h)?;
        let q = heads(&qkv.narrow(2, 0, HIDDEN)?)?;
        let k = heads(&qkv.narrow(2, HIDDEN, HIDDEN)?)?;
        let v = heads(&qkv.narrow(2, 2 * HIDDEN, HIDDEN)?)?;
        let (k, v) = self.cache.append(&k, &v)?;
        let x = (x + self.self_out.forward(&merge(&attention(&q, &k, &v)?)?)?)?;

        let h = self.cross_norm.forward(&x)?;
        let (ck, cv) = self.cross.as_ref().ok_or_else(|| anyhow!("decoder not started"))?;
        let q = heads(&self.cross_q.forward(&h)?)?;
        let x = (&x + self.cross_out.forward(&merge(&attention(&q, ck, cv)?)?)?)?;

        let h = self.ffn_norm.forward(&x)?;
        let h = self.fc2.forward(&self.fc1.forward(&h)?.gelu_erf()?)?;
        Ok((x + h)?)
    }
}

pub struct GpuDecoder {
    device: Device,
    dtype: DType,
    embed: Vec<Embedding>,
    positions: Tensor,
    layers: Vec<Layer>,
    norm: LayerNorm,
    /// The four codebook heads stacked: `[CODEBOOKS * VOCAB, HIDDEN]`.
    heads: Proj,
    enc_to_dec: Linear,
    position: usize,
}

/// Sinusoidal position table `[positions, dim]`, as `MusicgenSinusoidalPositionalEmbedding`: cos half, then sin half.
/// The checkpoint stores this table; the function documents and tests how it is built.
#[cfg_attr(not(test), allow(dead_code))]
fn sinusoids(positions: usize, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let scale = (10_000f64).ln() / (half - 1) as f64;
    let mut table = Vec::with_capacity(positions * dim);
    for p in 0..positions {
        let angles: Vec<f64> = (0..half).map(|i| p as f64 * (-(i as f64) * scale).exp()).collect();
        table.extend(angles.iter().map(|a| a.cos() as f32));
        table.extend(angles.iter().map(|a| a.sin() as f32));
    }
    Ok(Tensor::from_vec(table, (positions, dim), device)?)
}

impl GpuDecoder {
    /// Load the decoder. Matrix weights are quantized to 8 bits by default (`DRIFTSCAPE_QUANT=q8_0`); `q4k` quantizes
    /// further, `none` keeps them plain, in which case `DRIFTSCAPE_DTYPE=f16` (or `bf16`) runs at half precision.
    pub fn load(weights: &Path) -> Result<Self> {
        if !weights.exists() {
            anyhow::bail!("GPU decoder needs {}; download it from facebook/musicgen-small", weights.display());
        }
        let device = Device::metal_if_available(0)?;
        let quant = match std::env::var("DRIFTSCAPE_QUANT").as_deref() {
            Ok("none") => None,
            Ok("q4k") => Some(GgmlDType::Q4K),
            _ => Some(GgmlDType::Q8_0),
        };
        // Quantized kernels take f32 activations.
        let dtype = match (quant, std::env::var("DRIFTSCAPE_DTYPE").as_deref()) {
            (None, Ok("f16")) => DType::F16,
            (None, Ok("bf16")) => DType::BF16,
            _ => DType::F32,
        };
        // SAFETY: the file is memory-mapped read-only and not modified while the decoder lives.
        let map = |dtype, device: &Device| unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device) };
        let w = Weights {
            device_vb: map(dtype, &device).with_context(|| format!("loading {}", weights.display()))?,
            cpu_vb: map(DType::F32, &Device::Cpu)?,
            device: device.clone(),
            quant,
        };
        let vb = w.device_vb.clone();
        let dec = vb.pp("decoder.model.decoder");
        let dec_w = w.pp("decoder.model.decoder");
        let head_names: Vec<String> = (0..CODEBOOKS).map(|k| format!("lm_heads.{k}")).collect();
        let head_names: Vec<&str> = head_names.iter().map(String::as_str).collect();
        Ok(Self {
            embed: (0..CODEBOOKS)
                .map(|k| embedding(VOCAB + 1, HIDDEN, dec.pp(format!("embed_tokens.{k}"))))
                .collect::<candle_core::Result<_>>()?,
            positions: dec.get((MAX_POSITIONS, HIDDEN), "embed_positions.weights")?,
            layers: (0..LAYERS)
                .map(|l| Layer::load(dec_w.pp(format!("layers.{l}")), MAX_POSITIONS))
                .collect::<Result<_>>()?,
            norm: layer_norm(HIDDEN, EPS, dec.pp("layer_norm"))?,
            heads: w.pp("decoder").proj(&head_names, HIDDEN, VOCAB)?,
            enc_to_dec: linear(TEXT_DIM, HIDDEN, vb.pp("enc_to_dec_proj"))?,
            position: 0,
            device,
            dtype,
        })
    }

    /// Whether this decoder actually runs on a GPU (false when Metal is unavailable or not compiled in).
    pub fn on_gpu(&self) -> bool {
        !matches!(self.device, Device::Cpu)
    }
}

impl Decoder for GpuDecoder {
    fn start(&mut self, states: &[f32], mask: &[i64], seq: usize) -> Result<()> {
        let keep: Vec<f32> = mask.iter().map(|&m| m as f32).collect();
        let keep = Tensor::from_vec(keep, (2, seq, 1), &self.device)?.to_dtype(self.dtype)?;
        let states = Tensor::from_vec(states.to_vec(), (2, seq, TEXT_DIM), &self.device)?.to_dtype(self.dtype)?;
        let states = self.enc_to_dec.forward(&states)?.broadcast_mul(&keep)?;
        for layer in &mut self.layers {
            layer.start(&states, MAX_POSITIONS)?;
        }
        self.position = 0;
        Ok(())
    }

    fn step(&mut self, tokens: &[i64]) -> Result<Vec<f32>> {
        let logits = self.step_lazy(tokens)?;
        Ok(logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?)
    }
}

impl GpuDecoder {
    /// One step without reading the result back: the logits stay on the device. For profiling.
    pub fn step_lazy(&mut self, tokens: &[i64]) -> Result<Tensor> {
        // Sum the codebook embeddings per batch row: rows are batch-major, codebook-minor.
        let mut x = self.positions.narrow(0, self.position, 1)?.broadcast_as((2, HIDDEN))?.contiguous()?;
        for (k, embed) in self.embed.iter().enumerate() {
            let ids: Vec<u32> = (0..2).map(|b| tokens[b * CODEBOOKS + k] as u32).collect();
            x = (x + embed.forward(&Tensor::from_vec(ids, 2, &self.device)?)?)?;
        }
        let mut x = x.unsqueeze(1)?; // [2, 1, hidden]
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }
        self.position += 1;

        // [2, 1, codebooks * vocab]: batch-major, codebook-minor rows once flattened, like the tokens.
        Ok(self.heads.forward(&self.norm.forward(&x)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoids_match_musicgen() {
        let t = sinusoids(3, 8, &Device::Cpu).unwrap().to_vec2::<f32>().unwrap();
        // Position 0: cos(0) = 1 in the first half, sin(0) = 0 in the second.
        assert_eq!(t[0], vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        // Position 1, first frequency is 1 rad.
        assert!((t[1][0] - 1f32.cos()).abs() < 1e-6);
        assert!((t[1][4] - 1f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn cpu_attention_is_a_weighted_average() {
        // One query attending equally to two keys returns the mean of the values.
        let q = Tensor::zeros((1, 1, 1, HEAD_DIM), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones((1, 1, 2, HEAD_DIM), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::cat(
            &[
                Tensor::zeros((1, 1, 1, HEAD_DIM), DType::F32, &Device::Cpu).unwrap(),
                Tensor::ones((1, 1, 1, HEAD_DIM), DType::F32, &Device::Cpu).unwrap(),
            ],
            2,
        )
        .unwrap();
        let out = attention(&q, &k, &v).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(out.iter().all(|&x| (x - 0.5).abs() < 1e-6));
    }
}
