//! Tempo estimation from an onset-strength envelope.
//!
//! The clip is reduced to a 100 Hz envelope of rising energy (onsets). A tempo whose beat period lines the envelope up
//! with itself scores high. To avoid answering half the true tempo, each candidate is penalised by how well the
//! envelope lines up at half its period: if the music has pulses between our beats, the faster tempo is the beat.
//!
//! Ambient material often has no pulse at all. Then the best score is weak and [`estimate`] returns `None`, and the
//! caller should fall back to a length in seconds.

/// Envelope frames per second.
const FRAME_RATE: f64 = 100.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tempo {
    /// Beats per minute.
    pub bpm: f64,
    /// How strongly the envelope repeats at this tempo, from 0 (not at all) to 1 (perfectly).
    pub confidence: f64,
}

/// The range of tempos considered.
#[derive(Debug, Clone, Copy)]
pub struct Range {
    pub min_bpm: f64,
    pub max_bpm: f64,
}

impl Default for Range {
    fn default() -> Self {
        Self { min_bpm: 60.0, max_bpm: 180.0 }
    }
}

/// Below this confidence the clip is treated as having no steady pulse.
pub const MIN_CONFIDENCE: f64 = 0.15;

/// Estimate the tempo of a mono signal. Returns `None` for clips shorter than four seconds or without a steady pulse.
pub fn estimate(samples: &[f32], sample_rate: u32, range: Range) -> Option<Tempo> {
    let Envelope { onsets: envelope, flux, level } = onset_envelope(samples, sample_rate)?;
    if (envelope.len() as f64) < 4.0 * FRAME_RATE {
        return None;
    }
    // A steady tone still ripples a little frame to frame. Without real changes in level there is no pulse.
    if flux < MIN_FLUX * level {
        return None;
    }
    let zero = autocorrelation(&envelope, 0.0);
    if zero <= 0.0 {
        return None;
    }

    let mut best = Tempo { bpm: 0.0, confidence: f64::MIN };
    let mut bpm = range.min_bpm;
    while bpm <= range.max_bpm {
        let lag = 60.0 / bpm * FRAME_RATE;
        let half = autocorrelation(&envelope, lag / 2.0).max(0.0);
        let score = ((autocorrelation(&envelope, lag) - 0.5 * half) / zero).clamp(0.0, 1.0);
        if score > best.confidence {
            best = Tempo { bpm, confidence: score };
        }
        bpm += 0.1;
    }
    if best.confidence < MIN_CONFIDENCE {
        return None;
    }
    best.bpm = refine(&envelope, best.bpm);
    Some(best)
}

/// Sharpen a coarse tempo to 0.01 BPM. A one-beat lag is only a few dozen envelope frames, so it cannot tell 89.6 from
/// 90 BPM; but a loop cut at the wrong one drifts by tens of milliseconds every repeat. Lining the envelope up with
/// itself many beats later can: the error grows with every beat, so the summed match peaks sharply at the true period.
fn refine(envelope: &[f64], coarse_bpm: f64) -> f64 {
    let usable = envelope.len() as f64 * 0.6;
    let score = |bpm: f64| -> f64 {
        let lag = 60.0 / bpm * FRAME_RATE;
        (1..).map(|m| f64::from(m) * lag).take_while(|&l| l <= usable).map(|l| autocorrelation(envelope, l)).sum()
    };
    let mut best = (coarse_bpm, score(coarse_bpm));
    let mut step = -60;
    while step <= 60 {
        let bpm = coarse_bpm + f64::from(step) * 0.01;
        let s = score(bpm);
        if s > best.1 {
            best = (bpm, s);
        }
        step += 1;
    }
    (best.0 * 100.0).round() / 100.0
}

/// Mean onset strength must be at least this fraction of the mean level for a clip to have a pulse.
const MIN_FLUX: f64 = 0.01;

struct Envelope {
    /// Half-wave rectified first difference of the RMS envelope, mean removed, at [`FRAME_RATE`].
    onsets: Vec<f64>,
    /// Mean of the rectified difference before the mean was removed.
    flux: f64,
    /// Mean RMS.
    level: f64,
}

fn onset_envelope(samples: &[f32], sample_rate: u32) -> Option<Envelope> {
    let hop = (f64::from(sample_rate) / FRAME_RATE).round() as usize;
    // A 40 ms Hann-windowed frame spans several periods of even a low drone, so a steady tone gives a flat envelope.
    let frame = hop * 4;
    if hop == 0 || samples.len() < frame * 2 {
        return None;
    }
    let window: Vec<f64> =
        (0..frame).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / frame as f64).cos()).collect();
    let window_energy: f64 = window.iter().map(|w| w * w).sum();
    let rms: Vec<f64> = (0..=(samples.len() - frame) / hop)
        .map(|k| {
            let w = &samples[k * hop..k * hop + frame];
            let e: f64 = w.iter().zip(&window).map(|(&s, &h)| (f64::from(s) * h).powi(2)).sum();
            (e / window_energy).sqrt()
        })
        .collect();
    let level = rms.iter().sum::<f64>() / rms.len() as f64;
    let mut onsets: Vec<f64> = rms.windows(2).map(|w| (w[1] - w[0]).max(0.0)).collect();
    let flux = onsets.iter().sum::<f64>() / onsets.len() as f64;
    onsets.iter_mut().for_each(|v| *v -= flux);
    Some(Envelope { onsets, flux, level })
}

/// Mean of `x[k] * x[k + lag]` over the overlap, with `lag` fractional (linear interpolation).
fn autocorrelation(x: &[f64], lag: f64) -> f64 {
    let whole = lag.floor() as usize;
    let frac = lag - lag.floor();
    if whole + 1 >= x.len() {
        return 0.0;
    }
    let n = x.len() - whole - 1;
    let sum: f64 = (0..n).map(|k| x[k] * (x[k + whole] * (1.0 - frac) + x[k + whole + 1] * frac)).sum();
    sum / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Short decaying noise bursts on every beat, with an accent on the first of each bar.
    fn click_track(bpm: f64, seconds: f64, rate: u32) -> Vec<f32> {
        let n = (seconds * f64::from(rate)) as usize;
        let period = 60.0 / bpm * f64::from(rate);
        let mut out = vec![0.0_f32; n];
        let mut seed = 0x2545_f491_u32;
        let mut beat = 0;
        loop {
            let start = (beat as f64 * period) as usize;
            if start >= n {
                break;
            }
            let gain = if beat % 4 == 0 { 1.0 } else { 0.6 };
            for (i, s) in out[start..n.min(start + 2000)].iter_mut().enumerate() {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let noise = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
                *s = noise * gain * (-(i as f32) / 300.0).exp();
            }
            beat += 1;
        }
        out
    }

    #[test]
    fn finds_common_tempos() {
        for bpm in [72.0, 90.0, 120.0, 140.0] {
            let t = estimate(&click_track(bpm, 12.0, 32_000), 32_000, Range::default()).expect("a pulse");
            assert!((t.bpm - bpm).abs() <= 1.0, "expected {bpm}, got {t:?}");
        }
    }

    #[test]
    fn is_precise_enough_to_cut_bars() {
        // 0.1 BPM off at 90 BPM drifts about 12 ms over 4 bars; anything coarser is audible on repeat.
        for bpm in [84.0, 90.0, 123.0] {
            let t = estimate(&click_track(bpm, 20.0, 32_000), 32_000, Range::default()).expect("a pulse");
            assert!((t.bpm - bpm).abs() <= 0.1, "expected {bpm}, got {t:?}");
        }
    }

    #[test]
    fn gives_up_on_a_drone() {
        let drone: Vec<f32> =
            (0..32_000 * 8).map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 110.0 / 32_000.0).sin() * 0.5).collect();
        assert_eq!(estimate(&drone, 32_000, Range::default()), None);
    }

    #[test]
    fn needs_four_seconds() {
        assert_eq!(estimate(&click_track(120.0, 3.0, 32_000), 32_000, Range::default()), None);
    }
}
