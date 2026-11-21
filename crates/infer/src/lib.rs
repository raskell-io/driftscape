//! Driftscape voices: the models that turn a prompt into audio.
//!
//! A voice is anything that can generate a clip from a [`Request`]. MusicGen is the first; Stable Audio Open is
//! planned for realistic ambience. Everything around the model (looping, tempo, mixing) lives in `driftscape-core`.

pub mod musicgen;
mod sampling;

use driftscape_core::Audio;

/// What to generate.
#[derive(Debug, Clone)]
pub struct Request {
    /// The full text prompt, after any genre template has been applied.
    pub prompt: String,
    /// Length of the clip in seconds.
    pub seconds: f32,
    /// Classifier-free guidance: how strongly to follow the prompt. 3 is MusicGen's default.
    pub guidance: f32,
    /// Sampling temperature. Higher is more varied.
    pub temperature: f32,
    /// Sample only among this many most likely tokens.
    pub top_k: usize,
    /// Random seed; the same seed and request give the same clip.
    pub seed: u64,
}

impl Request {
    pub fn new(prompt: impl Into<String>, seconds: f32, seed: u64) -> Self {
        Self { prompt: prompt.into(), seconds, guidance: 3.0, temperature: 1.0, top_k: 250, seed }
    }
}

/// Generation progress, reported once per step.
#[derive(Debug, Clone, Copy)]
pub struct Progress {
    pub step: usize,
    pub total: usize,
}

/// A model that generates audio.
pub trait Voice {
    /// Generate a clip. `progress` is called after every step.
    fn generate(&mut self, request: &Request, progress: &mut dyn FnMut(Progress)) -> anyhow::Result<Audio>;
}
