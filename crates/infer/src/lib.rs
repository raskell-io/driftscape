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
    /// Continue from the end of an earlier clip: these tokens are replayed first, then `seconds` of new audio follow.
    /// The returned clip's audio covers both, so its start overlaps the earlier clip's end.
    pub continue_from: Option<Tokens>,
}

impl Request {
    pub fn new(prompt: impl Into<String>, seconds: f32, seed: u64) -> Self {
        Self { prompt: prompt.into(), seconds, guidance: 3.0, temperature: 1.0, top_k: 250, seed, continue_from: None }
    }
}

/// A voice's tokens for a clip: one sequence per codebook, all of equal length, at `frame_rate` frames per second.
#[derive(Debug, Clone, PartialEq)]
pub struct Tokens {
    pub codebooks: Vec<Vec<i64>>,
    pub frame_rate: f32,
}

impl Tokens {
    pub fn frames(&self) -> usize {
        self.codebooks.first().map_or(0, Vec::len)
    }

    /// The last `seconds` (at most the whole clip).
    pub fn tail(&self, seconds: f32) -> Tokens {
        let n = ((seconds * self.frame_rate).round() as usize).min(self.frames());
        let from = self.frames() - n;
        Tokens { codebooks: self.codebooks.iter().map(|c| c[from..].to_vec()).collect(), frame_rate: self.frame_rate }
    }
}

/// A generated clip.
#[derive(Debug, Clone)]
pub struct Clip {
    pub audio: Audio,
    pub tokens: Tokens,
    /// How many seconds at the start of `audio` replay [`Request::continue_from`]; 0 for a fresh clip.
    pub overlap_seconds: f32,
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
    fn generate(&mut self, request: &Request, progress: &mut dyn FnMut(Progress)) -> anyhow::Result<Clip>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_takes_the_end_of_every_codebook() {
        let t = Tokens { codebooks: vec![(0..100).collect(), (100..200).collect()], frame_rate: 50.0 };
        let tail = t.tail(0.5);
        assert_eq!(tail.frames(), 25);
        assert_eq!(tail.codebooks[0][0], 75);
        assert_eq!(tail.codebooks[1][24], 199);
        assert_eq!(t.tail(10.0).frames(), 100);
    }
}
