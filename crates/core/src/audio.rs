//! The sample buffer every part of the engine passes around.

/// Audio as one `Vec<f32>` per channel (planar), all of equal length, nominally in `-1.0..=1.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f32>>,
}

impl Audio {
    /// A buffer from planar channels. Panics if there are no channels or their lengths differ, since every caller
    /// builds these from known-good data; [`crate::wav::decode`] validates untrusted input before getting here.
    pub fn new(sample_rate: u32, channels: Vec<Vec<f32>>) -> Self {
        assert!(!channels.is_empty(), "audio needs at least one channel");
        let len = channels[0].len();
        assert!(channels.iter().all(|c| c.len() == len), "channels differ in length");
        assert!(sample_rate > 0, "sample rate must be positive");
        Self { sample_rate, channels }
    }

    /// A single-channel buffer.
    pub fn mono(sample_rate: u32, samples: Vec<f32>) -> Self {
        Self::new(sample_rate, vec![samples])
    }

    /// Frames per channel.
    pub fn len(&self) -> usize {
        self.channels[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Duration in seconds.
    pub fn seconds(&self) -> f64 {
        self.len() as f64 / f64::from(self.sample_rate)
    }

    /// The average of all channels, used for analysis.
    pub fn mixdown(&self) -> Vec<f32> {
        if self.channels.len() == 1 {
            return self.channels[0].clone();
        }
        let n = self.channels.len() as f32;
        (0..self.len()).map(|i| self.channels.iter().map(|c| c[i]).sum::<f32>() / n).collect()
    }

    /// The largest absolute sample value.
    pub fn peak(&self) -> f32 {
        self.channels.iter().flatten().fold(0.0_f32, |m, s| m.max(s.abs()))
    }

    /// Scale so the peak sits at `target` (linear, e.g. 0.89 for about -1 dBFS). Silence stays silent.
    pub fn normalize(&mut self, target: f32) {
        let peak = self.peak();
        if peak > 0.0 {
            let gain = target / peak;
            self.channels.iter_mut().flatten().for_each(|s| *s *= gain);
        }
    }

    /// Frames `start..end` of every channel as a new buffer.
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self::new(self.sample_rate, self.channels.iter().map(|c| c[start..end].to_vec()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixdown_averages_channels() {
        let a = Audio::new(10, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(a.mixdown(), vec![0.5, 0.5]);
    }

    #[test]
    fn normalize_hits_target_peak() {
        let mut a = Audio::mono(10, vec![0.1, -0.5, 0.25]);
        a.normalize(1.0);
        assert!((a.peak() - 1.0).abs() < 1e-6);
        assert!((a.channels[0][0] - 0.2).abs() < 1e-6);
    }
}
