//! A small real-time mixer for live mode: buffers scheduled at absolute frame times, faded in and out.
//!
//! The audio callback owns a [`Mixer`] and calls [`Mixer::render`]. A control thread decides what plays when and sends
//! [`Command`]s (over a lock-free queue, in the CLI). Rendering never allocates or frees: voices live in a fixed set of
//! slots, and finished buffers are handed back through [`Mixer::take_finished`] so the control thread drops them.
//!
//! Time is counted in output frames since the mixer started. Commands name the frame they take effect at, so the
//! control thread can schedule ahead and the transition lands on an exact frame regardless of callback timing.

use std::sync::Arc;

/// Mono samples at the output rate, shared between the control thread and the mixer.
pub type Buffer = Arc<[f32]>;

/// Most voices that can sound at once. Live mode needs two during a crossfade; the rest is headroom.
pub const SLOTS: usize = 8;

#[derive(Debug, Clone)]
pub enum Command {
    /// Start playing `buffer` from frame `offset` at time `at`, fading in over `fade_in` frames. `id` names the voice
    /// for later commands.
    Play { id: u64, buffer: Buffer, offset: usize, at: u64, fade_in: usize },
    /// Fade voice `id` out over `fade` frames starting at time `at`; it stops after that.
    FadeOut { id: u64, at: u64, fade: usize },
}

#[derive(Debug)]
struct Voice {
    id: u64,
    buffer: Buffer,
    /// Frame in `buffer` that plays at time `start`.
    offset: usize,
    start: u64,
    fade_in: usize,
    fade_out: Option<(u64, usize)>,
}

impl Voice {
    /// The voice's gain at absolute time `t`, or `None` once it has finished.
    fn sample(&self, t: u64) -> Option<f32> {
        if t < self.start {
            return Some(0.0);
        }
        let elapsed = (t - self.start) as usize;
        let i = self.offset + elapsed;
        if i >= self.buffer.len() {
            return None;
        }
        let mut gain = 1.0;
        if elapsed < self.fade_in {
            gain *= equal_power_in(elapsed, self.fade_in);
        }
        if let Some((at, fade)) = self.fade_out
            && t >= at
        {
            let into = (t - at) as usize;
            if into >= fade {
                return None;
            }
            // The exact complement of the fade-in curve, so a crossfade keeps constant power.
            gain *= equal_power_in(fade - 1 - into, fade);
        }
        Some(self.buffer[i] * gain)
    }
}

/// Rising equal-power gain at step `i` of `n`: sin of a quarter turn, so a crossfade keeps the loudness of unrelated
/// material. Over an exactly repeated sound (the continuation overlap) it bulges by up to 3 dB mid-fade; for a few
/// seconds of ambient sound that is unnoticeable, and it is the right choice when the overlap has drifted.
fn equal_power_in(i: usize, n: usize) -> f32 {
    ((i as f32 + 0.5) / n as f32 * std::f32::consts::FRAC_PI_2).sin()
}

#[derive(Debug, Default)]
pub struct Mixer {
    voices: [Option<Voice>; SLOTS],
    finished: [Option<Buffer>; SLOTS],
    time: u64,
}

impl Mixer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Frames rendered so far: the current time.
    pub fn time(&self) -> u64 {
        self.time
    }

    /// Apply a command. Returns `false` if a `Play` found no free slot (the command is dropped).
    pub fn apply(&mut self, command: Command) -> bool {
        match command {
            Command::Play { id, buffer, offset, at, fade_in } => match self.voices.iter_mut().find(|v| v.is_none()) {
                Some(slot) => {
                    *slot = Some(Voice { id, buffer, offset, start: at, fade_in, fade_out: None });
                    true
                }
                None => false,
            },
            Command::FadeOut { id, at, fade } => {
                for v in self.voices.iter_mut().flatten().filter(|v| v.id == id) {
                    v.fade_out = Some((at, fade.max(1)));
                }
                true
            }
        }
    }

    /// Mix the next `out.len()` frames into `out` (overwriting it).
    pub fn render(&mut self, out: &mut [f32]) {
        out.fill(0.0);
        for slot in &mut self.voices {
            let Some(voice) = slot else { continue };
            let mut done = false;
            for (n, o) in out.iter_mut().enumerate() {
                match voice.sample(self.time + n as u64) {
                    Some(s) => *o += s,
                    None => {
                        done = true;
                        break;
                    }
                }
            }
            if done && let Some(v) = slot.take() {
                // Hand the buffer back instead of dropping it here: freeing memory has no time bound.
                if let Some(free) = self.finished.iter_mut().find(|f| f.is_none()) {
                    *free = Some(v.buffer);
                } else {
                    // No room to hand it back: keep the voice (silent, finished) until the control thread collects.
                    *slot = Some(v);
                }
            }
        }
        self.time += out.len() as u64;
    }

    /// Buffers of voices that finished since the last call; drop them outside the audio callback.
    pub fn take_finished(&mut self) -> impl Iterator<Item = Buffer> + '_ {
        self.finished.iter_mut().filter_map(Option::take)
    }

    /// Ids of voices still sounding or scheduled.
    pub fn active(&self) -> impl Iterator<Item = u64> + '_ {
        self.voices.iter().flatten().map(|v| v.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant(value: f32, len: usize) -> Buffer {
        vec![value; len].into()
    }

    #[test]
    fn plays_from_the_scheduled_frame_and_offset() {
        let mut m = Mixer::new();
        let ramp: Buffer = (0..100).map(|i| i as f32).collect::<Vec<_>>().into();
        m.apply(Command::Play { id: 1, buffer: ramp, offset: 10, at: 5, fade_in: 0 });
        let mut out = [0.0; 8];
        m.render(&mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn crossfade_keeps_power_and_hands_back_the_old_buffer() {
        let mut m = Mixer::new();
        m.apply(Command::Play { id: 1, buffer: constant(1.0, 1000), offset: 0, at: 0, fade_in: 0 });
        m.apply(Command::FadeOut { id: 1, at: 100, fade: 200 });
        m.apply(Command::Play { id: 2, buffer: constant(1.0, 1000), offset: 0, at: 100, fade_in: 200 });
        let mut out = vec![0.0; 400];
        m.render(&mut out);
        assert!(out[..100].iter().all(|&s| s == 1.0));
        // Equal power: the summed gains peak at √2 mid-fade on identical material, and never dip below 1.
        assert!(out[100..300].iter().all(|&s| (1.0..=1.4143).contains(&s)), "{:?}", &out[195..205]);
        assert!(out[300..].iter().all(|&s| (s - 1.0).abs() < 1e-6));
        assert_eq!(m.take_finished().count(), 1);
        assert_eq!(m.active().collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn voices_end_with_their_buffer() {
        let mut m = Mixer::new();
        m.apply(Command::Play { id: 1, buffer: constant(0.5, 3), offset: 0, at: 0, fade_in: 0 });
        let mut out = [9.0; 5];
        m.render(&mut out);
        assert_eq!(out, [0.5, 0.5, 0.5, 0.0, 0.0]);
        assert_eq!(m.active().count(), 0);
    }

    #[test]
    fn full_mixer_refuses_more_voices() {
        let mut m = Mixer::new();
        for id in 0..SLOTS as u64 {
            assert!(m.apply(Command::Play { id, buffer: constant(0.0, 10), offset: 0, at: 0, fade_in: 0 }));
        }
        assert!(!m.apply(Command::Play { id: 99, buffer: constant(0.0, 10), offset: 0, at: 0, fade_in: 0 }));
    }
}

/// A slow automatic level control for the mixed output.
///
/// Generated music sometimes writes its own ending: a long fade to near silence, then a new phrase that starts loud.
/// For an always-on stream both are unwelcome. The leveler follows the loudness over a few seconds and steers the gain
/// toward a target, within bounds, so quiet stretches are lifted and sudden jumps are eased in, while the music keeps
/// most of its own dynamics. A soft limiter keeps peaks below full scale.
#[derive(Debug, Clone)]
pub struct Leveler {
    target_rms: f32,
    min_gain: f32,
    max_gain: f32,
    /// Smoothed mean square of the input.
    power: f32,
    gain: f32,
    power_coef: f32,
    gain_coef: f32,
}

impl Leveler {
    /// `seconds` is the time constant of both the loudness measurement and the gain movement.
    pub fn new(sample_rate: u32, target_rms: f32, seconds: f32) -> Self {
        let coef = (-1.0 / (seconds * sample_rate as f32)).exp();
        Self {
            target_rms,
            min_gain: 0.5,
            max_gain: 4.0,
            power: target_rms * target_rms,
            gain: 1.0,
            power_coef: coef,
            gain_coef: coef,
        }
    }

    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples {
            self.power = self.power_coef * self.power + (1.0 - self.power_coef) * *s * *s;
            // Below this the input is treated as silence, so the gain does not chase a stopped stream.
            let rms = self.power.sqrt().max(self.target_rms / (2.0 * self.max_gain));
            let wanted = (self.target_rms / rms).clamp(self.min_gain, self.max_gain);
            self.gain = self.gain_coef * self.gain + (1.0 - self.gain_coef) * wanted;
            *s = soft_limit(*s * self.gain);
        }
    }
}

/// Unity below 0.8, then a smooth curve that approaches but never reaches 1.0.
pub(crate) fn soft_limit(x: f32) -> f32 {
    const KNEE: f32 = 0.8;
    let a = x.abs();
    if a <= KNEE {
        return x;
    }
    let over = (a - KNEE) / (1.0 - KNEE);
    (KNEE + (1.0 - KNEE) * over.tanh()).copysign(x)
}

#[cfg(test)]
mod leveler_tests {
    use super::*;

    fn tone(amplitude: f32, len: usize) -> Vec<f32> {
        (0..len).map(|i| (i as f32 * 0.05).sin() * amplitude).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn lifts_a_quiet_stretch_toward_the_target() {
        let mut l = Leveler::new(1000, 0.1, 1.0);
        let mut x = tone(0.014, 20_000); // RMS ≈ 0.01, ten times too quiet
        l.process(&mut x);
        let late = rms(&x[15_000..]);
        assert!((0.035..=0.045).contains(&late), "boost is capped at 4×: {late}");
    }

    #[test]
    fn eases_a_sudden_jump_and_never_clips() {
        let mut l = Leveler::new(1000, 0.1, 1.0);
        let mut quiet = tone(0.014, 10_000);
        l.process(&mut quiet);
        let mut loud = tone(0.9, 10_000);
        l.process(&mut loud);
        assert!(loud.iter().all(|s| s.abs() <= 1.0), "never beyond full scale");
        assert!(rms(&loud[9_000..]) < rms(&loud[..500]), "the gain comes back down after the jump");
    }

    #[test]
    fn soft_limit_is_transparent_below_the_knee() {
        assert_eq!(soft_limit(0.5), 0.5);
        assert!(soft_limit(3.0) <= 1.0 && soft_limit(3.0) > 0.99);
        assert!(soft_limit(0.9) < 0.9 && soft_limit(0.9) > 0.85, "smooth above the knee");
        assert_eq!(soft_limit(-3.0), -soft_limit(3.0));
    }
}
