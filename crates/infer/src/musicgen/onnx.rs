//! The decoder on ONNX Runtime (CPU), using `decoder_model_merged_q4.onnx` from `Xenova/musicgen-small`.

use super::{CODEBOOKS, Decoder, TEXT_DIM, VOCAB};
use anyhow::{Result, anyhow};
use ort::{
    environment::Environment,
    session::{Session, SessionInputValue},
    value::{DynValue, Tensor},
};
use std::{borrow::Cow, path::Path};

const LAYERS: usize = 24;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;

pub struct OnnxDecoder {
    session: Session,
    /// `(past input name, present output name)` for every layer's self- and cross-attention keys and values.
    cache_names: Vec<(String, String)>,
    states: Option<Tensor<f32>>,
    mask: Option<Tensor<i64>>,
    cache: Vec<DynValue>,
}

impl OnnxDecoder {
    pub fn load(env: &Environment, path: &Path) -> Result<Self> {
        let cache_names = (0..LAYERS)
            .flat_map(|l| {
                ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]
                    .map(|part| (format!("past_key_values.{l}.{part}"), format!("present.{l}.{part}")))
            })
            .collect();
        Ok(Self { session: super::session(env, path)?, cache_names, states: None, mask: None, cache: Vec::new() })
    }
}

impl Decoder for OnnxDecoder {
    fn start(&mut self, states: &[f32], mask: &[i64], seq: usize) -> Result<()> {
        self.states = Some(Tensor::from_array(([2, seq, TEXT_DIM], states.to_vec()))?);
        self.mask = Some(Tensor::from_array(([2, seq], mask.to_vec()))?);
        self.cache.clear();
        Ok(())
    }

    fn step(&mut self, tokens: &[i64]) -> Result<Vec<f32>> {
        let (states, mask) =
            self.states.as_ref().zip(self.mask.as_ref()).ok_or_else(|| anyhow!("decoder not started"))?;
        let first = self.cache.is_empty();
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
            ("input_ids".into(), Tensor::from_array(([2 * CODEBOOKS, 1], tokens.to_vec()))?.into()),
            ("encoder_hidden_states".into(), states.into()),
            ("encoder_attention_mask".into(), mask.into()),
            ("use_cache_branch".into(), Tensor::from_array(([1], vec![!first]))?.into()),
        ];
        if first {
            // Empty caches; on this first pass the decoder computes self- and cross-attention keys itself.
            for (past, _) in &self.cache_names {
                let empty = Tensor::<f32>::from_array(([2, HEADS, 0, HEAD_DIM], Vec::new()))?;
                inputs.push((past.as_str().into(), empty.into()));
            }
        } else {
            for ((past, _), value) in self.cache_names.iter().zip(&self.cache) {
                inputs.push((past.as_str().into(), value.into()));
            }
        }

        let mut outputs = self.session.run(inputs)?;
        let logits = {
            let (shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
            let positions = shape[1] as usize;
            (0..2 * CODEBOOKS)
                .flat_map(|row| logits[(row * positions + positions - 1) * VOCAB..][..VOCAB].iter().copied())
                .collect()
        };

        // Self-attention keys grow every step. Cross-attention keys depend only on the text, so they come from the
        // first pass; on cached passes the merged decoder returns placeholders for them.
        let mut previous = std::mem::take(&mut self.cache).into_iter();
        let mut updated = Vec::with_capacity(self.cache_names.len());
        for (_, present) in &self.cache_names {
            let value = match previous.next() {
                Some(v) if !present.contains("decoder") => v,
                _ => outputs.remove(present).ok_or_else(|| anyhow!("decoder did not return {present}"))?,
            };
            updated.push(value);
        }
        drop(outputs);
        self.cache = updated;
        Ok(logits)
    }
}
