//! Time-stretching without changing pitch: WSOLA (waveform-similarity overlap-add).
//!
//! The output is built from short windowed frames of the input, laid down at a fixed hop. Each frame is read from
//! around where the stretch ratio says it should come from, nudged within a small tolerance to the spot whose waveform
//! best continues the previous frame, so the overlap-add stays in phase. That keeps transients like kick drums
//! intact, which matters for fitting beats to a tempo. Ratios close to 1 (a few percent, as when correcting a
//! generated clip's tempo) are essentially transparent.

use crate::Audio;

/// Frame length in seconds. Long enough to hold a period of low notes, short enough to keep transients tight.
const FRAME_SECONDS: f64 = 0.04;

/// Stretch `audio` so it lasts `ratio` times as long (ratio > 1 is slower, < 1 faster). Pitch is unchanged.
/// All channels use the same frame positions, so stereo stays aligned.
pub fn stretch(audio: &Audio, ratio: f64) -> Audio {
    assert!(ratio > 0.0, "stretch ratio must be positive");
    let frame = ((FRAME_SECONDS * f64::from(audio.sample_rate)) as usize).max(16) & !1;
    let hop_out = frame / 2;
    let hop_in = hop_out as f64 / ratio;
    let tolerance = frame / 4;
    let window: Vec<f32> =
        (0..frame).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / frame as f32).cos()).collect();

    let len = audio.len();
    let out_len = (len as f64 * ratio).round() as usize;
    if len < frame + 2 * tolerance || (ratio - 1.0).abs() < 1e-9 {
        return audio.clone();
    }
    let mono = audio.mixdown();
    let frames = out_len / hop_out + 1;
    let mut positions = Vec::with_capacity(frames);
    // Where the previous frame was read from; the "natural" continuation is one output hop after it.
    let mut previous: Option<usize> = None;
    let last_start = len - frame;
    for k in 0..frames {
        let nominal = ((k as f64 * hop_in).round() as usize).min(last_start);
        let chosen = match previous {
            None => nominal,
            Some(p) => {
                let natural = (p + hop_out).min(last_start);
                let target = &mono[natural..natural + frame];
                let lo = nominal.saturating_sub(tolerance);
                let hi = (nominal + tolerance).min(last_start);
                (lo..=hi)
                    .step_by(2)
                    .map(|s| (s, correlation(&mono[s..s + frame], target)))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map_or(nominal, |(s, _)| s)
            }
        };
        positions.push(chosen);
        previous = Some(chosen);
    }

    let channels = audio
        .channels
        .iter()
        .map(|x| {
            let mut out = vec![0.0f32; out_len + frame];
            let mut norm = vec![0.0f32; out_len + frame];
            for (k, &pos) in positions.iter().enumerate() {
                let at = k * hop_out;
                for i in 0..frame {
                    out[at + i] += x[pos + i] * window[i];
                    norm[at + i] += window[i];
                }
            }
            out.truncate(out_len);
            // Hann windows at half overlap sum to 1 except at the very edges; divide there to keep the level.
            for (o, n) in out.iter_mut().zip(&norm) {
                if *n > 1e-3 {
                    *o /= n;
                }
            }
            out
        })
        .collect();
    Audio::new(audio.sample_rate, channels)
}

/// Unnormalised cross-correlation; the energy of both windows is similar at nearby positions, so normalising adds
/// cost without changing the winner much.
fn correlation(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tempo;

    fn sine(rate: u32, freq: f32, seconds: f32) -> Audio {
        let n = (rate as f32 * seconds) as usize;
        Audio::mono(
            rate,
            (0..n).map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin() * 0.5).collect(),
        )
    }

    /// Rising zero crossings per second, a robust pitch measure for a sine.
    fn frequency(a: &Audio) -> f32 {
        let x = &a.channels[0];
        let rising = x.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        rising as f32 / a.seconds() as f32
    }

    #[test]
    fn changes_length_but_not_pitch() {
        let a = sine(16_000, 220.0, 3.0);
        for ratio in [0.95, 1.05, 1.2] {
            let b = stretch(&a, ratio);
            assert_eq!(b.len(), (a.len() as f64 * ratio).round() as usize);
            let f = frequency(&b);
            assert!((f - 220.0).abs() < 3.0, "ratio {ratio}: pitch moved to {f}");
        }
    }

    #[test]
    fn keeps_the_level() {
        let a = sine(16_000, 220.0, 3.0);
        let b = stretch(&a, 1.07);
        let rms = |x: &[f32]| (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt();
        let (ra, rb) = (rms(&a.channels[0]), rms(&b.channels[0][1000..b.len() - 1000]));
        assert!((ra - rb).abs() / ra < 0.05, "{ra} vs {rb}");
    }

    #[test]
    fn moves_a_beat_to_the_new_tempo() {
        // Clicks at 120 BPM, stretched by 120/114: the beat must land at 114 BPM.
        let rate = 16_000;
        let period = rate as usize / 2;
        let mut x = vec![0.0f32; rate as usize * 16];
        for b in 0..32 {
            for i in 0..400 {
                x[b * period + i] = ((i * 7919 % 13) as f32 / 6.5 - 1.0) * (-(i as f32) / 80.0).exp();
            }
        }
        let a = Audio::mono(rate, x);
        let b = stretch(&a, 120.0 / 114.0);
        let t = tempo::estimate(&b.channels[0], rate, tempo::Range::default()).expect("a pulse");
        assert!((t.bpm - 114.0).abs() < 0.3, "got {t:?}");
    }

    #[test]
    fn identity_and_short_input_pass_through() {
        let a = sine(16_000, 220.0, 1.0);
        assert_eq!(stretch(&a, 1.0), a);
        let tiny = Audio::mono(16_000, vec![0.1; 100]);
        assert_eq!(stretch(&tiny, 1.1).len(), 100);
    }
}
