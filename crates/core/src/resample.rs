//! Sample-rate conversion with a windowed-sinc filter.
//!
//! MusicGen produces 32 kHz audio; DAWs and sound cards mostly run at 44.1 or 48 kHz. This is a direct (not
//! polyphase) implementation: slower than it could be, but exact for any ratio and plenty fast for clips of seconds.

use crate::Audio;
use std::f64::consts::PI;

/// Zero crossings of the sinc on each side of the centre. 32 gives a steep low-pass with negligible ripple.
const HALF_TAPS: f64 = 32.0;

/// Resample `audio` to `to_rate`. Returns a clone when the rate already matches.
pub fn resample(audio: &Audio, to_rate: u32) -> Audio {
    if audio.sample_rate == to_rate || audio.is_empty() {
        return Audio::new(to_rate, audio.channels.clone());
    }
    let from = f64::from(audio.sample_rate);
    let to = f64::from(to_rate);
    let step = from / to; // input samples per output sample
    // When downsampling, lower the cutoff to the new Nyquist frequency so nothing aliases.
    let cutoff = (to / from).min(1.0);
    let reach = HALF_TAPS / cutoff; // filter half-width in input samples
    let out_len = (audio.len() as f64 * to / from).round() as usize;

    let channels = audio
        .channels
        .iter()
        .map(|x| {
            (0..out_len)
                .map(|i| {
                    let t = i as f64 * step;
                    let lo = ((t - reach).ceil().max(0.0)) as usize;
                    let hi = ((t + reach).floor() as usize).min(x.len() - 1);
                    let mut acc = 0.0;
                    for (j, &s) in x.iter().enumerate().take(hi + 1).skip(lo) {
                        let d = t - j as f64;
                        acc += f64::from(s) * cutoff * sinc(cutoff * d) * blackman(d / reach);
                    }
                    acc as f32
                })
                .collect()
        })
        .collect();
    Audio::new(to_rate, channels)
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 { 1.0 } else { (PI * x).sin() / (PI * x) }
}

/// Blackman window over `-1.0..=1.0`, zero outside.
fn blackman(u: f64) -> f64 {
    if u.abs() >= 1.0 {
        return 0.0;
    }
    let p = PI * (u + 1.0); // maps -1..1 onto 0..2π
    0.42 - 0.5 * p.cos() + 0.08 * (2.0 * p).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, freq: f64, seconds: f64) -> Audio {
        let n = (f64::from(rate) * seconds) as usize;
        Audio::mono(rate, (0..n).map(|i| (2.0 * PI * freq * i as f64 / f64::from(rate)).sin() as f32).collect())
    }

    #[test]
    fn keeps_a_sine_intact_when_upsampling() {
        let out = resample(&sine(32_000, 440.0, 0.5), 48_000);
        assert_eq!(out.sample_rate, 48_000);
        assert_eq!(out.len(), 24_000);
        let expected = sine(48_000, 440.0, 0.5);
        // Ignore the filter's edge ramps.
        let err = out.channels[0][1000..23_000]
            .iter()
            .zip(&expected.channels[0][1000..23_000])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(err < 1e-3, "max error {err}");
    }

    #[test]
    fn removes_content_above_the_new_nyquist() {
        // 20 kHz cannot exist at 32 kHz; it must be filtered out, not folded down to 12 kHz.
        let out = resample(&sine(48_000, 20_000.0, 0.25), 32_000);
        let rms = (out.channels[0][500..7500].iter().map(|s| s * s).sum::<f32>() / 7000.0).sqrt();
        assert!(rms < 0.01, "aliased energy {rms}");
    }
}
