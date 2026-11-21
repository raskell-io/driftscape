//! Producer mode's arrangement: layers (tracks) of clips placed on a timeline, and the mixdown that turns them into
//! one stereo buffer. The browser preview, the WAV export and `driftscape render` all use [`render`], so what plays is
//! what gets written.
//!
//! Clips refer to their audio by index into the `sources` slice passed to [`render`]; the arrangement itself holds no
//! samples, so it is small enough to save and undo freely.

use crate::{Audio, mixer::soft_limit, resample::resample};
use serde::{Deserialize, Serialize};

/// The whole piece.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Arrangement {
    /// The mixdown's sample rate. Sources at other rates are resampled.
    pub sample_rate: u32,
    pub tracks: Vec<Track>,
}

/// A layer: clips that share a gain, a pan position and a mute switch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Track {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub gain_db: f32,
    /// -1 is hard left, 0 the centre, 1 hard right.
    #[serde(default)]
    pub pan: f32,
    #[serde(default)]
    pub mute: bool,
    pub clips: Vec<Placed>,
}

/// A stretch of a source placed on the timeline. Times are in seconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Placed {
    /// Index into the sources given to [`render`].
    pub source: usize,
    /// Where the clip starts on the timeline.
    pub at: f64,
    /// Where in the source it starts playing (trims the source's start).
    #[serde(default)]
    pub offset: f64,
    /// How long it plays; `None` plays to the source's end.
    #[serde(default)]
    pub length: Option<f64>,
    #[serde(default)]
    pub fade_in: f64,
    #[serde(default)]
    pub fade_out: f64,
    #[serde(default)]
    pub curve: Curve,
    #[serde(default)]
    pub gain_db: f32,
}

/// The shape of a clip's fades. Two overlapping clips with matching fades crossfade at constant loudness: equal power
/// for unrelated material, linear for material that lines up sample for sample (an extension that replays the end of
/// the clip before it).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Curve {
    #[default]
    EqualPower,
    Linear,
}

impl Curve {
    /// Gain at `x` from 0 (silent) to 1 (full).
    fn at(self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        match self {
            Curve::EqualPower => (x * std::f32::consts::FRAC_PI_2).sin(),
            Curve::Linear => x,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrangeError {
    /// A clip names a source that wasn't given.
    UnknownSource(usize),
}

impl std::fmt::Display for ArrangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrangeError::UnknownSource(i) => write!(f, "a clip refers to source {i}, which doesn't exist"),
        }
    }
}

impl std::error::Error for ArrangeError {}

impl Placed {
    /// How long the clip plays, given its source's duration.
    pub fn duration(&self, source_seconds: f64) -> f64 {
        let rest = (source_seconds - self.offset).max(0.0);
        self.length.map_or(rest, |l| l.clamp(0.0, rest))
    }
}

impl Arrangement {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate, tracks: Vec::new() }
    }

    /// Where the last clip ends, in seconds (muted tracks included, so muting doesn't change the length).
    pub fn seconds(&self, sources: &[Audio]) -> Result<f64, ArrangeError> {
        let mut end: f64 = 0.0;
        for clip in self.tracks.iter().flat_map(|t| &t.clips) {
            let source = sources.get(clip.source).ok_or(ArrangeError::UnknownSource(clip.source))?;
            end = end.max(clip.at.max(0.0) + clip.duration(source.seconds()));
        }
        Ok(end)
    }
}

fn db(gain_db: f32) -> f32 {
    10f32.powf(gain_db / 20.0)
}

/// Mix the arrangement down to stereo at its sample rate. Mono sources are panned at constant power; stereo sources
/// are balanced. The sum goes through the same soft limiter as live mode, so it never clips.
pub fn render(arrangement: &Arrangement, sources: &[Audio]) -> Result<Audio, ArrangeError> {
    let rate = arrangement.sample_rate;
    let frames = (arrangement.seconds(sources)? * rate as f64).round() as usize;
    let mut out = [vec![0.0f32; frames], vec![0.0f32; frames]];

    // Resample each source at most once, and only if a clip on an audible track uses it.
    let mut converted: Vec<Option<Audio>> = vec![None; sources.len()];
    for track in arrangement.tracks.iter().filter(|t| !t.mute) {
        for clip in &track.clips {
            let source = &sources[clip.source];
            if source.sample_rate != rate && converted[clip.source].is_none() {
                converted[clip.source] = Some(resample(source, rate));
            }
        }
    }

    for track in arrangement.tracks.iter().filter(|t| !t.mute) {
        let pan = track.pan.clamp(-1.0, 1.0);
        for clip in &track.clips {
            let source = converted[clip.source].as_ref().unwrap_or(&sources[clip.source]);
            let gain = db(track.gain_db) * db(clip.gain_db);
            // Per-channel gains: constant-power pan for mono, balance for stereo.
            let (gl, gr) = if source.channel_count() == 1 {
                let theta = (pan + 1.0) * std::f32::consts::FRAC_PI_4;
                (theta.cos(), theta.sin())
            } else {
                ((1.0 - pan).min(1.0), (1.0 + pan).min(1.0))
            };
            let left = &source.channels[0];
            let right = &source.channels[source.channel_count().min(2) - 1];

            let start = (clip.at.max(0.0) * rate as f64).round() as usize;
            let from = ((clip.offset.max(0.0) * rate as f64).round() as usize).min(source.len());
            let len = ((clip.duration(source.seconds()) * rate as f64).round() as usize)
                .min(source.len() - from)
                .min(frames.saturating_sub(start));
            let fade_in = (clip.fade_in.max(0.0) * rate as f64).round() as usize;
            let fade_out = (clip.fade_out.max(0.0) * rate as f64).round() as usize;

            for i in 0..len {
                let mut g = gain;
                if i < fade_in {
                    g *= clip.curve.at(i as f32 / fade_in as f32);
                }
                let left_to_end = len - i;
                if left_to_end <= fade_out {
                    g *= clip.curve.at((left_to_end - 1) as f32 / fade_out as f32);
                }
                out[0][start + i] += left[from + i] * g * gl;
                out[1][start + i] += right[from + i] * g * gr;
            }
        }
    }

    for channel in &mut out {
        for s in channel.iter_mut() {
            *s = soft_limit(*s);
        }
    }
    let [l, r] = out;
    Ok(Audio::new(rate, vec![l, r]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 8000;

    fn sine(freq: f32, seconds: f32, amp: f32) -> Audio {
        let n = (seconds * RATE as f32) as usize;
        Audio::mono(RATE, (0..n).map(|i| amp * (i as f32 * freq * std::f32::consts::TAU / RATE as f32).sin()).collect())
    }

    fn noise(seed: u64, seconds: f32, amp: f32) -> Audio {
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let n = (seconds * RATE as f32) as usize;
        Audio::mono(
            RATE,
            (0..n)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    amp * (((x >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0)
                })
                .collect(),
        )
    }

    fn clip(source: usize, at: f64) -> Placed {
        Placed {
            source,
            at,
            offset: 0.0,
            length: None,
            fade_in: 0.0,
            fade_out: 0.0,
            curve: Curve::EqualPower,
            gain_db: 0.0,
        }
    }

    fn track(clips: Vec<Placed>) -> Track {
        Track { name: String::new(), gain_db: 0.0, pan: 0.0, mute: false, clips }
    }

    fn rms(s: &[f32]) -> f32 {
        (s.iter().map(|x| x * x).sum::<f32>() / s.len().max(1) as f32).sqrt()
    }

    #[test]
    fn places_clips_and_sets_the_length() {
        let sources = [sine(440.0, 1.0, 0.5)];
        let mut a = Arrangement::new(RATE);
        a.tracks.push(track(vec![clip(0, 0.5), clip(0, 2.0)]));
        let out = render(&a, &sources).unwrap();
        assert_eq!(out.channel_count(), 2);
        assert_eq!(out.len(), 3 * RATE as usize);
        let l = &out.channels[0];
        assert_eq!(rms(&l[..RATE as usize / 2]), 0.0, "silent before the first clip");
        assert!(rms(&l[RATE as usize..3 * RATE as usize / 2]) > 0.2, "sound inside the first clip");
        assert_eq!(rms(&l[3 * RATE as usize / 2..2 * RATE as usize]), 0.0, "silent in the gap");
    }

    #[test]
    fn mono_in_the_centre_is_equal_on_both_sides_and_pan_moves_it() {
        let sources = [sine(440.0, 1.0, 0.5)];
        let mut a = Arrangement::new(RATE);
        a.tracks.push(track(vec![clip(0, 0.0)]));
        let out = render(&a, &sources).unwrap();
        assert_eq!(out.channels[0], out.channels[1]);
        let centre = rms(&out.channels[0]);

        a.tracks[0].pan = -1.0;
        let out = render(&a, &sources).unwrap();
        assert!(rms(&out.channels[1]) < 1e-4, "hard left leaves the right silent");
        // Constant power: one side at full pan carries the power both sides had in the centre.
        let hard = rms(&out.channels[0]);
        assert!((hard * hard - 2.0 * centre * centre).abs() < 1e-3);
    }

    #[test]
    fn fades_start_and_end_at_silence_without_a_click() {
        let sources = [sine(440.0, 2.0, 0.5)];
        let mut a = Arrangement::new(RATE);
        let mut c = clip(0, 0.0);
        c.fade_in = 0.5;
        c.fade_out = 0.5;
        a.tracks.push(track(vec![c]));
        let l = render(&a, &sources).unwrap().channels[0].clone();
        assert!(l[0].abs() < 1e-6 && l[l.len() - 1].abs() < 1e-6);
        // No jump bigger than the sine's own largest step anywhere.
        let max_step = 0.5 * (440.0 * std::f32::consts::TAU / RATE as f32);
        assert!(l.windows(2).all(|w| (w[1] - w[0]).abs() <= max_step * 1.01));
    }

    #[test]
    fn equal_power_crossfade_of_unrelated_clips_keeps_the_level() {
        let sources = [noise(1, 4.0, 0.3), noise(2, 4.0, 0.3)];
        let mut a = Arrangement::new(RATE);
        let mut first = clip(0, 0.0);
        first.fade_out = 2.0;
        let mut second = clip(1, 2.0);
        second.fade_in = 2.0;
        a.tracks.push(track(vec![first, second]));
        let l = render(&a, &sources).unwrap().channels[0].clone();
        let before = rms(&l[RATE as usize / 2..RATE as usize * 3 / 2]);
        let block = RATE as usize / 4;
        for k in 8..16 {
            let level = rms(&l[k * block..(k + 1) * block]);
            let db = 20.0 * (level / before).log10();
            assert!(db.abs() < 1.0, "block {k}: {db:.2} dB");
        }
    }

    #[test]
    fn linear_crossfade_of_aligned_material_is_seamless() {
        // An extension replays the end of the clip before it: both carry the same samples in the overlap.
        let whole = sine(220.0, 6.0, 0.5);
        let head = Audio::mono(RATE, whole.channels[0][..4 * RATE as usize].to_vec());
        let tail = Audio::mono(RATE, whole.channels[0][3 * RATE as usize..].to_vec());
        let mut a = Arrangement::new(RATE);
        let mut first = clip(0, 0.0);
        first.fade_out = 1.0;
        first.curve = Curve::Linear;
        let mut second = clip(1, 3.0);
        second.fade_in = 1.0;
        second.curve = Curve::Linear;
        a.tracks.push(track(vec![first, second]));
        let out = render(&a, &[head, tail]).unwrap();
        let reference = &whole.channels[0];
        let centre = std::f32::consts::FRAC_1_SQRT_2;
        let worst = out.channels[0].iter().zip(reference).map(|(o, r)| (o - r * centre).abs()).fold(0.0, f32::max);
        assert!(worst < 2e-3, "differs from the uncut sine by {worst}");
    }

    #[test]
    fn trims_gains_mutes_and_resamples() {
        let sources = [sine(440.0, 2.0, 0.25), Audio::mono(16000, vec![0.25; 16000])];
        let mut a = Arrangement::new(RATE);
        let mut trimmed = clip(0, 0.0);
        trimmed.offset = 0.5;
        trimmed.length = Some(1.0);
        trimmed.gain_db = 6.0;
        a.tracks.push(track(vec![trimmed]));
        a.tracks.push(track(vec![clip(1, 0.0)]));
        let out = render(&a, &sources).unwrap();
        assert_eq!(out.len(), RATE as usize, "the 16 kHz source lasts 1 s at 8 kHz too");

        a.tracks[1].mute = true;
        let out = render(&a, &sources).unwrap();
        let expected = 0.25 * db(6.0) * std::f32::consts::FRAC_1_SQRT_2 * std::f32::consts::FRAC_1_SQRT_2;
        assert!((rms(&out.channels[0]) - expected).abs() < 0.01);
    }

    #[test]
    fn a_missing_source_is_an_error_and_json_round_trips() {
        let mut a = Arrangement::new(RATE);
        a.tracks.push(track(vec![clip(3, 0.0)]));
        assert_eq!(render(&a, &[]), Err(ArrangeError::UnknownSource(3)));

        let json = r#"{"sample_rate": 32000, "tracks": [{"clips": [{"source": 0, "at": 1.5, "curve": "linear"}]}]}"#;
        let parsed: Arrangement = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.tracks[0].clips[0].curve, Curve::Linear);
        assert_eq!(parsed.tracks[0].clips[0].length, None);
        let again: Arrangement = serde_json::from_str(&serde_json::to_string(&parsed).unwrap()).unwrap();
        assert_eq!(again, parsed);
    }
}
