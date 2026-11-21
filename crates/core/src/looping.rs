//! Seamless loops.
//!
//! A clip of length `L` repeats cleanly only if what comes after its end sounds like its start. Generated clips never
//! do. So we take `L + F` samples, where `F` is a crossfade length, and fold the extra tail into the head: the first
//! `F` samples of the loop fade from the tail into the head. When playback wraps from the last sample back to the
//! first, it continues exactly where the original audio went on.
//!
//! Where the loop starts matters too: [`make_loop`] searches the clip for the start whose audio matches the audio
//! one loop length later, so the crossfade blends two similar sounds instead of smearing two different ones.

use crate::Audio;

/// How to cut the loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopSpec {
    /// Loop length in frames.
    pub length: usize,
    /// Crossfade length in frames; the clip must be at least `length + crossfade` long.
    pub crossfade: usize,
}

impl LoopSpec {
    /// A loop of whole bars at a tempo, e.g. 8 bars of 4/4 at 84 BPM.
    pub fn bars(bars: u32, beats_per_bar: u32, bpm: f64, sample_rate: u32, crossfade_seconds: f64) -> Self {
        let beats = f64::from(bars * beats_per_bar);
        Self {
            length: (beats * 60.0 / bpm * f64::from(sample_rate)).round() as usize,
            crossfade: (crossfade_seconds * f64::from(sample_rate)).round() as usize,
        }
    }

    /// A loop of a length in seconds.
    pub fn seconds(seconds: f64, sample_rate: u32, crossfade_seconds: f64) -> Self {
        Self {
            length: (seconds * f64::from(sample_rate)).round() as usize,
            crossfade: (crossfade_seconds * f64::from(sample_rate)).round() as usize,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoopError {
    /// The clip is shorter than `length + crossfade`.
    TooShort {
        needed: usize,
        available: usize,
    },
    EmptyLoop,
}

impl std::fmt::Display for LoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { needed, available } => {
                write!(f, "clip has {available} frames but the loop and its crossfade need {needed}")
            }
            Self::EmptyLoop => write!(f, "loop length must be greater than zero"),
        }
    }
}

impl std::error::Error for LoopError {}

/// A loop cut from a clip, with where it came from.
#[derive(Debug, Clone)]
pub struct Loop {
    pub audio: Audio,
    /// The frame in the source clip where the loop starts.
    pub start: usize,
    /// How similar the audio at the start and one loop length later is, from -1 to 1. Above about 0.5 the seam is
    /// usually inaudible even before the crossfade.
    pub seam_similarity: f32,
}

/// Cut a seamless loop from `clip`.
pub fn make_loop(clip: &Audio, spec: LoopSpec) -> Result<Loop, LoopError> {
    if spec.length == 0 {
        return Err(LoopError::EmptyLoop);
    }
    let needed = spec.length + spec.crossfade;
    if clip.len() < needed {
        return Err(LoopError::TooShort { needed, available: clip.len() });
    }

    let (start, seam_similarity) = best_start(clip, spec);
    let channels = clip
        .channels
        .iter()
        .map(|x| {
            let mut out = x[start..start + spec.length].to_vec();
            for (i, s) in out.iter_mut().enumerate().take(spec.crossfade) {
                let (fade_in, fade_out) = equal_power(i, spec.crossfade);
                *s = *s * fade_in + x[start + spec.length + i] * fade_out;
            }
            out
        })
        .collect();
    Ok(Loop { audio: Audio::new(clip.sample_rate, channels), start, seam_similarity })
}

/// Join `next` onto the end of `current`, crossfading over `crossfade` frames: the last `crossfade` frames of
/// `current` overlap the first `crossfade` frames of `next`. Live mode stitches each new segment on this way.
pub fn crossfade_join(current: &Audio, next: &Audio, crossfade: usize) -> Audio {
    assert_eq!(current.sample_rate, next.sample_rate, "sample rates differ");
    assert_eq!(current.channel_count(), next.channel_count(), "channel counts differ");
    let crossfade = crossfade.min(current.len()).min(next.len());
    let head = current.len() - crossfade;
    let channels = current
        .channels
        .iter()
        .zip(&next.channels)
        .map(|(a, b)| {
            let mut out = Vec::with_capacity(head + b.len());
            out.extend_from_slice(&a[..head]);
            for i in 0..crossfade {
                let (fade_in, fade_out) = equal_power(i, crossfade);
                out.push(a[head + i] * fade_out + b[i] * fade_in);
            }
            out.extend_from_slice(&b[crossfade..]);
            out
        })
        .collect();
    Audio::new(current.sample_rate, channels)
}

/// Where to jump back to when a clip has to repeat.
///
/// Playback is about to cross from `from` onwards; the result is an earlier frame `s` in `earliest..=latest` whose
/// next `window` frames sound most like `samples[from..from + window]`, so a crossfade from `from` to `s` is smooth.
/// With `bar` (frames per bar), only starts a whole number of bars before `from` are considered, so a beat keeps its
/// place. Returns the frame and its similarity (-1 to 1); `None` when no candidate fits.
pub fn rejoin_point(
    samples: &[f32],
    from: usize,
    window: usize,
    earliest: usize,
    latest: usize,
    bar: Option<usize>,
) -> Option<(usize, f32)> {
    if from + window > samples.len() || earliest > latest {
        return None;
    }
    let target = &samples[from..from + window];
    let latest = latest.min(from.saturating_sub(window));
    let score = |s: usize| similarity(&samples[s..s + window], target);
    let pick = |candidates: &mut dyn Iterator<Item = usize>| {
        candidates.map(|s| (s, score(s))).max_by(|a, b| a.1.total_cmp(&b.1))
    };
    match bar {
        Some(bar) if bar > 0 => {
            pick(&mut (1..=from / bar).map(|n| from - n * bar).filter(|s| (earliest..=latest).contains(s)))
        }
        _ => {
            const STEP: usize = 64;
            let (coarse, _) = pick(&mut (earliest..=latest).step_by(STEP))?;
            pick(&mut (coarse.saturating_sub(STEP).max(earliest)..=(coarse + STEP).min(latest)))
        }
    }
}

/// Gains for step `i` of `n` in an equal-power crossfade: (rising, falling), with rising² + falling² = 1, so the
/// blend of two unrelated sounds keeps its loudness.
fn equal_power(i: usize, n: usize) -> (f32, f32) {
    let t = (i as f32 + 0.5) / n as f32 * std::f32::consts::FRAC_PI_2;
    (t.sin(), t.cos())
}

/// The start in `0..=len - length - crossfade` whose surroundings best match the audio one loop length later.
/// A coarse pass every [`COARSE_STEP`] frames, then a fine pass around the winner.
fn best_start(clip: &Audio, spec: LoopSpec) -> (usize, f32) {
    const COARSE_STEP: usize = 64;
    let mono = clip.mixdown();
    let last = clip.len() - spec.length - spec.crossfade;
    // Compare a window spanning the crossfade (at least 50 ms) so the match covers what will be blended.
    // Near the end the window is clipped so it never reads past the clip.
    let window = spec.crossfade.max(clip.sample_rate as usize / 20);
    let score = |s: usize| -> f32 {
        let end = (s + window).min(clip.len() - spec.length);
        similarity(&mono[s..end], &mono[s + spec.length..end + spec.length])
    };

    let mut best = (0, f32::MIN);
    for s in (0..=last).step_by(COARSE_STEP) {
        let v = score(s);
        if v > best.1 {
            best = (s, v);
        }
    }
    for s in best.0.saturating_sub(COARSE_STEP)..=(best.0 + COARSE_STEP).min(last) {
        let v = score(s);
        if v > best.1 {
            best = (s, v);
        }
    }
    best
}

/// Normalised cross-correlation of two equal-length windows, in -1..=1. Silence against silence counts as a match.
fn similarity(a: &[f32], b: &[f32]) -> f32 {
    let (mut ab, mut aa, mut bb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa < 1e-12 && bb < 1e-12 {
        return 1.0;
    }
    (ab / (aa.sqrt() * bb.sqrt()).max(1e-12)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(rate: u32, freq: f32, frames: usize) -> Audio {
        Audio::mono(rate, (0..frames).map(|i| (2.0 * PI * freq * i as f32 / rate as f32).sin() * 0.8).collect())
    }

    /// The jump between the last and first sample, relative to the largest step inside the loop.
    fn wrap_jump(a: &Audio) -> f32 {
        let x = &a.channels[0];
        let inner = x.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0_f32, f32::max);
        (x[0] - x[x.len() - 1]).abs() / inner
    }

    #[test]
    fn wraps_without_a_click() {
        // 1000 frames is not a whole number of 37 Hz periods, so a plain cut clicks.
        let clip = sine(8000, 37.0, 4000);
        let naive = clip.slice(0, 1000);
        assert!(wrap_jump(&naive) > 5.0, "the naive cut should click: {}", wrap_jump(&naive));

        let l = make_loop(&clip, LoopSpec { length: 1000, crossfade: 200 }).unwrap();
        assert_eq!(l.audio.len(), 1000);
        assert!(wrap_jump(&l.audio) <= 1.5, "seam jump {}", wrap_jump(&l.audio));
    }

    #[test]
    fn finds_the_matching_start() {
        // The clip changes timbre halfway; only starts in the first part line up with the audio a loop length later
        // if the loop length is a whole number of periods there. Both halves are periodic, so any start works, but
        // similarity at the best start must be near perfect.
        let clip = sine(8000, 40.0, 6000); // period 200 frames
        let l = make_loop(&clip, LoopSpec { length: 2000, crossfade: 100 }).unwrap();
        assert!(l.seam_similarity > 0.99, "similarity {}", l.seam_similarity);
    }

    #[test]
    fn bars_compute_length() {
        let spec = LoopSpec::bars(8, 4, 120.0, 48_000, 0.25);
        assert_eq!(spec.length, 16 * 48_000);
        assert_eq!(spec.crossfade, 12_000);
    }

    #[test]
    fn rejects_short_clips() {
        let clip = sine(8000, 40.0, 1000);
        let err = make_loop(&clip, LoopSpec { length: 1000, crossfade: 10 }).unwrap_err();
        assert_eq!(err, LoopError::TooShort { needed: 1010, available: 1000 });
    }

    #[test]
    fn rejoin_finds_the_same_phase_earlier() {
        // A 50-frame period: the best place to jump back to from frame 3000 is a whole number of periods earlier.
        let clip = sine(8000, 160.0, 4000);
        let (s, sim) = rejoin_point(&clip.channels[0], 3000, 200, 0, 2500, None).unwrap();
        assert_eq!((3000 - s) % 50, 0, "jumped to {s}");
        assert!(sim > 0.99);
    }

    #[test]
    fn rejoin_on_bars_only_lands_on_bar_lines() {
        let clip = sine(8000, 160.0, 4000);
        let (s, _) = rejoin_point(&clip.channels[0], 3000, 200, 0, 2500, Some(700)).unwrap();
        assert_eq!((3000 - s) % 700, 0);
        assert!(rejoin_point(&clip.channels[0], 3000, 200, 2600, 2700, Some(700)).is_none());
    }

    #[test]
    fn join_keeps_length_and_continuity() {
        let a = Audio::mono(100, vec![1.0; 50]);
        let b = Audio::mono(100, vec![1.0; 50]);
        let j = crossfade_join(&a, &b, 20);
        assert_eq!(j.len(), 80);
        // Equal-power on identical signals bumps loudness by at most sqrt(2) in the middle.
        assert!(j.channels[0].iter().all(|&s| (1.0..=1.42).contains(&s)));
    }
}

/// A loop fitted to a tempo: stretched from the tempo the clip was played at to the target, then cut to whole bars.
#[derive(Debug, Clone)]
pub struct Fitted {
    pub looped: Loop,
    /// The clip's own tempo, as given or detected.
    pub played_bpm: f64,
    /// How much the clip was stretched (played / target).
    pub ratio: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FitError {
    /// The clip has no steady pulse to measure; pass its tempo explicitly.
    NoPulse,
    /// The clip's tempo is too far from the target to stretch cleanly (more than ±25 %).
    TooFar {
        played_bpm: f64,
        target_bpm: f64,
    },
    Loop(LoopError),
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPulse => write!(f, "the clip has no steady pulse to measure; give its tempo explicitly"),
            Self::TooFar { played_bpm, target_bpm } => {
                write!(f, "the clip plays at {played_bpm} BPM, too far from {target_bpm} BPM to stretch cleanly")
            }
            Self::Loop(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for FitError {}

/// Fit `clip` to `bars` bars of `beats_per_bar` at `target_bpm`. The clip's own tempo is `played_bpm` if given,
/// otherwise detected; a detection at half or double speed is folded to the nearest octave of the target.
pub fn fit_loop(
    clip: &Audio,
    target_bpm: f64,
    bars: u32,
    beats_per_bar: u32,
    crossfade_seconds: f64,
    played_bpm: Option<f64>,
) -> Result<Fitted, FitError> {
    let detected =
        || crate::tempo::estimate(&clip.mixdown(), clip.sample_rate, crate::tempo::Range::default()).map(|t| t.bpm);
    let mut played = played_bpm.or_else(detected).ok_or(FitError::NoPulse)?;
    while played < target_bpm / std::f64::consts::SQRT_2 {
        played *= 2.0;
    }
    while played > target_bpm * std::f64::consts::SQRT_2 {
        played /= 2.0;
    }
    let ratio = played / target_bpm;
    if !(0.8..=1.25).contains(&ratio) {
        return Err(FitError::TooFar { played_bpm: played, target_bpm });
    }
    let stretched = if (ratio - 1.0).abs() < 1e-4 { clip.clone() } else { crate::stretch::stretch(clip, ratio) };
    let spec = LoopSpec::bars(bars, beats_per_bar, target_bpm, clip.sample_rate, crossfade_seconds);
    let looped = make_loop(&stretched, spec).map_err(FitError::Loop)?;
    Ok(Fitted { looped, played_bpm: played, ratio })
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    fn clicks(bpm: f64, seconds: f64, rate: u32) -> Audio {
        let n = (seconds * f64::from(rate)) as usize;
        let period = 60.0 / bpm * f64::from(rate);
        let mut x = vec![0.0f32; n];
        let mut b = 0;
        while ((b as f64) * period) < n as f64 {
            let s = (b as f64 * period) as usize;
            for i in 0..400.min(n - s) {
                x[s + i] = ((i * 7919 % 13) as f32 / 6.5 - 1.0) * (-(i as f32) / 80.0).exp();
            }
            b += 1;
        }
        Audio::mono(rate, x)
    }

    #[test]
    fn fits_a_slightly_slow_clip_to_exact_bars() {
        let clip = clicks(116.0, 20.0, 16_000);
        let fit = fit_loop(&clip, 118.0, 4, 4, 0.1, None).unwrap();
        assert!((fit.played_bpm - 116.0).abs() < 0.2, "{}", fit.played_bpm);
        // Exactly 4 bars at 118 BPM.
        assert_eq!(fit.looped.audio.len(), (16.0 * 60.0 / 118.0 * 16_000.0f64).round() as usize);
        let t = crate::tempo::estimate(&fit.looped.audio.channels[0], 16_000, crate::tempo::Range::default());
        assert!((t.unwrap().bpm - 118.0).abs() < 0.5);
    }

    #[test]
    fn folds_a_half_speed_detection() {
        let clip = clicks(118.0, 20.0, 16_000);
        let fit = fit_loop(&clip, 118.0, 2, 4, 0.1, Some(59.0)).unwrap();
        assert!((fit.ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn refuses_a_tempo_too_far_away() {
        let clip = clicks(90.0, 20.0, 16_000);
        assert!(matches!(fit_loop(&clip, 120.0, 2, 4, 0.1, Some(90.0)), Err(FitError::TooFar { .. })));
    }
}
