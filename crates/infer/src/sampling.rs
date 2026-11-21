//! Token sampling: a small seeded PRNG and top-k sampling with temperature.

/// SplitMix64: tiny, fast and good enough for sampling. Seeded so a request is reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0.0..1.0`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sample an index from `logits`: keep the `top_k` largest, divide by `temperature`, softmax, draw.
/// `scratch` is reused between calls to avoid allocating every step.
pub fn top_k(logits: &[f32], top_k: usize, temperature: f32, rng: &mut Rng, scratch: &mut Vec<(f32, usize)>) -> usize {
    scratch.clear();
    scratch.extend(logits.iter().copied().zip(0..));
    let k = top_k.clamp(1, scratch.len());
    if k < scratch.len() {
        scratch.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        scratch.truncate(k);
    }
    let max = scratch.iter().map(|c| c.0).fold(f32::NEG_INFINITY, f32::max);
    let temperature = f64::from(temperature.max(1e-4));
    let weights: Vec<f64> = scratch.iter().map(|c| ((f64::from(c.0 - max)) / temperature).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut target = rng.next_f64() * total;
    for (w, c) in weights.iter().zip(scratch.iter()) {
        target -= w;
        if target <= 0.0 {
            return c.1;
        }
    }
    scratch.last().map_or(0, |c| c.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_one_is_argmax() {
        let mut rng = Rng::new(1);
        let mut scratch = Vec::new();
        assert_eq!(top_k(&[0.1, 3.0, 2.0, -1.0], 1, 1.0, &mut rng, &mut scratch), 1);
    }

    #[test]
    fn never_samples_outside_top_k() {
        let mut rng = Rng::new(7);
        let mut scratch = Vec::new();
        let logits = [5.0, 4.9, 0.0, 0.0, 0.0, 0.0];
        for _ in 0..1000 {
            assert!(top_k(&logits, 2, 1.0, &mut rng, &mut scratch) < 2);
        }
    }

    #[test]
    fn same_seed_same_draws() {
        let (mut a, mut b) = (Rng::new(42), Rng::new(42));
        assert!((0..100).all(|_| a.next_u64() == b.next_u64()));
    }
}
