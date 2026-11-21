//! Procedural ambience beds: rain, wind, waves and fire, synthesised from noise. They need no model, so they run
//! anywhere the core does (natively, in the plugin, as wasm), never repeat, and sit under the music as the
//! "realistic" layer MusicGen can't make.
//!
//! Each [`Bed`] streams stereo audio block by block, allocation-free after construction, so it can run inside an
//! audio callback. The output depends only on the kind, the sample rate and the seed: any block sizes give the same
//! samples.
//!
//! The recipes follow the usual physical-modelling shortcuts (Farnell, *Designing Sound*):
//! - **rain**: a pink hiss for the distant downpour plus individual drops: short bursts through damped resonators at
//!   random pitches and positions.
//! - **wind**: pink noise through band-pass filters whose centre and level follow slow, smoothed random gusts.
//! - **waves**: brown noise swelling and receding over 7–13 s, low-passed so the wash brightens as each wave breaks,
//!   with a hiss of foam on the break.
//! - **fire**: a low, flickering roar plus crackles: sparse clicks in clusters, and the odd lower pop.

use crate::Audio;
use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    Rain,
    Wind,
    Waves,
    Fire,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Rain, Kind::Wind, Kind::Waves, Kind::Fire];

    pub fn id(self) -> &'static str {
        match self {
            Kind::Rain => "rain",
            Kind::Wind => "wind",
            Kind::Waves => "waves",
            Kind::Fire => "fire",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Rain => "Rain",
            Kind::Wind => "Wind",
            Kind::Waves => "Waves",
            Kind::Fire => "Fire",
        }
    }

    pub fn from_id(id: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.id() == id)
    }

    /// For the C ABI: the index in [`Kind::ALL`].
    pub fn from_index(i: u32) -> Option<Kind> {
        Kind::ALL.get(i as usize).copied()
    }
}

/// The bed a description calls for, if it names one: "rain on a tin roof" → rain, "campfire" → fire. The first
/// match in the text wins, so "wind and rain" is wind.
pub fn suggest(text: &str) -> Option<Kind> {
    const WORDS: &[(&str, Kind)] = &[
        ("rain", Kind::Rain),
        ("drizzle", Kind::Rain),
        ("downpour", Kind::Rain),
        ("shower", Kind::Rain),
        ("monsoon", Kind::Rain),
        ("wind", Kind::Wind),
        ("breeze", Kind::Wind),
        ("gust", Kind::Wind),
        ("gale", Kind::Wind),
        ("blizzard", Kind::Wind),
        ("desert", Kind::Wind),
        ("tundra", Kind::Wind),
        ("ocean", Kind::Waves),
        ("sea", Kind::Waves),
        ("wave", Kind::Waves),
        ("surf", Kind::Waves),
        ("beach", Kind::Waves),
        ("shore", Kind::Waves),
        ("harbo", Kind::Waves),
        ("coast", Kind::Waves),
        ("tide", Kind::Waves),
        ("storm", Kind::Rain),
        ("thunder", Kind::Rain),
        ("fire", Kind::Fire),
        ("campfire", Kind::Fire),
        ("bonfire", Kind::Fire),
        ("wildfire", Kind::Fire),
        ("hearth", Kind::Fire),
        ("ember", Kind::Fire),
        ("crackl", Kind::Fire),
        ("tavern", Kind::Fire),
    ];
    // ASCII only: the stems are English, and Unicode case tables would triple the wasm module.
    let text = text.to_ascii_lowercase();
    let mut best: Option<(usize, Kind)> = None;
    for word in text.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()) {
        let at = word.as_ptr() as usize - text.as_ptr() as usize;
        if let Some(&(_, kind)) = WORDS.iter().find(|(stem, _)| word.starts_with(stem)) {
            // "seafront" and "seashore" start with "sea"; "season" and "search" do not mean the sea.
            if word.starts_with("sea")
                && (word.starts_with("season") || word.starts_with("seat") || word.starts_with("search"))
            {
                continue;
            }
            if best.is_none_or(|(b, _)| at < b) {
                best = Some((at, kind));
            }
        }
    }
    best.map(|(_, k)| k)
}

/// A small, fast, deterministic random source (xorshift64*).
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform in [-1, 1).
    fn white(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// Pink noise (-3 dB per octave), Paul Kellet's economy filter over white noise.
#[derive(Debug, Clone, Default)]
struct Pink {
    b: [f32; 3],
}

impl Pink {
    fn next(&mut self, white: f32) -> f32 {
        self.b[0] = 0.99765 * self.b[0] + white * 0.0990460;
        self.b[1] = 0.96300 * self.b[1] + white * 0.2965164;
        self.b[2] = 0.57000 * self.b[2] + white * 1.0526913;
        (self.b[0] + self.b[1] + self.b[2] + white * 0.1848) * 0.25
    }
}

/// Brown noise (-6 dB per octave): leaky integrated white noise.
#[derive(Debug, Clone, Default)]
struct Brown(f32);

impl Brown {
    fn next(&mut self, white: f32) -> f32 {
        self.0 = (0.998 * self.0 + 0.02 * white).clamp(-1.0, 1.0);
        self.0 * 3.5
    }
}

/// One-pole low-pass.
#[derive(Debug, Clone, Default)]
struct Lowpass {
    a: f32,
    y: f32,
}

impl Lowpass {
    fn new(cutoff: f32, rate: f32) -> Self {
        let mut f = Lowpass::default();
        f.set(cutoff, rate);
        f
    }
    fn set(&mut self, cutoff: f32, rate: f32) {
        self.a = 1.0 - (-TAU * cutoff / rate).exp();
    }
    fn next(&mut self, x: f32) -> f32 {
        self.y += self.a * (x - self.y);
        self.y
    }
}

/// One-pole high-pass (the input minus its low-passed self).
#[derive(Debug, Clone, Default)]
struct Highpass(Lowpass);

impl Highpass {
    fn new(cutoff: f32, rate: f32) -> Self {
        Highpass(Lowpass::new(cutoff, rate))
    }
    fn next(&mut self, x: f32) -> f32 {
        x - self.0.next(x)
    }
}

/// RBJ band-pass biquad (constant 0 dB peak gain).
#[derive(Debug, Clone, Default)]
struct Bandpass {
    b0: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Bandpass {
    fn new(centre: f32, q: f32, rate: f32) -> Self {
        let mut f = Bandpass::default();
        f.set(centre, q, rate);
        f
    }
    fn set(&mut self, centre: f32, q: f32, rate: f32) {
        let w = TAU * centre.min(rate * 0.45) / rate;
        let alpha = w.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        self.b0 = alpha / a0;
        self.b2 = -alpha / a0;
        self.a1 = -2.0 * w.cos() / a0;
        self.a2 = (1.0 - alpha) / a0;
    }
    fn next(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// A slowly wandering value in [0, 1]: random targets, approached smoothly. Gusts, flicker, swells.
#[derive(Debug, Clone)]
struct Drift {
    value: f32,
    target: f32,
    /// Samples until a new target.
    left: u32,
    /// Seconds between targets, at most.
    every: f32,
    smooth: Lowpass,
}

impl Drift {
    fn new(every: f32, smoothing_hz: f32, rate: f32, rng: &mut Rng) -> Self {
        let start = rng.unit();
        let mut smooth = Lowpass::new(smoothing_hz, rate);
        smooth.y = start;
        Drift { value: start, target: start, left: 0, every, smooth }
    }
    fn next(&mut self, rate: f32, rng: &mut Rng) -> f32 {
        if self.left == 0 {
            self.target = rng.unit();
            self.left = (rng.range(0.3, 1.0) * self.every * rate) as u32;
        }
        self.left -= 1;
        self.value = self.smooth.next(self.target);
        self.value
    }
}

/// A struck resonator: a decaying sine from an impulse, for drops, crackles and pops.
#[derive(Debug, Clone, Copy, Default)]
struct Ping {
    y1: f32,
    y2: f32,
    c: f32,
    r2: f32,
    gain_l: f32,
    gain_r: f32,
    /// Samples left before it is inaudible.
    life: u32,
}

impl Ping {
    fn strike(freq: f32, decay_seconds: f32, amp: f32, pan: f32, rate: f32) -> Self {
        let r = (-1.0 / (decay_seconds * rate)).exp();
        let w = TAU * freq.min(rate * 0.45) / rate;
        let theta = (pan.clamp(-1.0, 1.0) + 1.0) * PI / 4.0;
        Ping {
            // An impulse into y = 2r cos(w) y1 - r² y2 + x, with the first output already taken: y1 = amp·sin(w).
            y1: amp * w.sin(),
            y2: 0.0,
            c: 2.0 * r * w.cos(),
            r2: r * r,
            gain_l: theta.cos(),
            gain_r: theta.sin(),
            life: (decay_seconds * rate * 7.0) as u32,
        }
    }
    fn next(&mut self) -> f32 {
        let y = self.c * self.y1 - self.r2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        self.life = self.life.saturating_sub(1);
        y
    }
}

/// A fixed pool of pings; new strikes replace the quietest-to-be (oldest) when it's full. Never allocates.
#[derive(Debug, Clone)]
struct Pings {
    slots: [Ping; 48],
}

impl Pings {
    fn new() -> Self {
        Pings { slots: [Ping::default(); 48] }
    }
    fn strike(&mut self, ping: Ping) {
        let slot = self.slots.iter_mut().min_by_key(|p| p.life).expect("non-empty pool");
        *slot = ping;
    }
    fn next(&mut self) -> (f32, f32) {
        let (mut l, mut r) = (0.0, 0.0);
        for p in self.slots.iter_mut().filter(|p| p.life > 0) {
            let y = p.next();
            l += y * p.gain_l;
            r += y * p.gain_r;
        }
        (l, r)
    }
}

/// A Poisson process: whether an event happens this sample, at `per_second` on average.
fn event(rng: &mut Rng, per_second: f32, rate: f32) -> bool {
    rng.unit() < per_second / rate
}

#[derive(Debug, Clone)]
enum Voice {
    Rain { pink: [Pink; 2], hiss_hp: [Highpass; 2], hiss_lp: [Lowpass; 2], drops: Pings, heaviness: Drift },
    Wind { pink: [Pink; 2], band: [Bandpass; 2], body: [Lowpass; 2], gust: Drift, swirl: [Drift; 2] },
    Waves { brown: [Brown; 2], wash: [Lowpass; 2], foam_hp: [Highpass; 2], phase: f32, period: f32, delay: f32 },
    Fire { brown: [Brown; 2], roar: [Lowpass; 2], flicker: Drift, crackles: Pings, burst: u32 },
}

/// A streaming ambience generator.
#[derive(Debug, Clone)]
pub struct Bed {
    kind: Kind,
    rate: f32,
    rng: Rng,
    voice: Voice,
    /// 0..=1: how much of it there is (rain density, wind strength, wave size, fire size).
    intensity: f32,
}

impl Bed {
    pub fn new(kind: Kind, sample_rate: u32, seed: u64) -> Self {
        let rate = sample_rate as f32;
        let mut rng = Rng::new(seed ^ (kind as u64 + 1).wrapping_mul(0xA24B_AED4_963E_E407));
        let voice = match kind {
            Kind::Rain => Voice::Rain {
                pink: Default::default(),
                hiss_hp: [Highpass::new(500.0, rate), Highpass::new(520.0, rate)],
                hiss_lp: [Lowpass::new(7000.0, rate), Lowpass::new(6800.0, rate)],
                drops: Pings::new(),
                heaviness: Drift::new(12.0, 0.08, rate, &mut rng),
            },
            Kind::Wind => Voice::Wind {
                pink: Default::default(),
                band: [Bandpass::new(500.0, 1.2, rate), Bandpass::new(540.0, 1.2, rate)],
                body: [Lowpass::new(180.0, rate), Lowpass::new(170.0, rate)],
                gust: Drift::new(6.0, 0.25, rate, &mut rng),
                swirl: [Drift::new(3.0, 0.6, rate, &mut rng), Drift::new(3.0, 0.6, rate, &mut rng)],
            },
            Kind::Waves => {
                let period = rng.range(7.0, 13.0);
                Voice::Waves {
                    brown: Default::default(),
                    wash: [Lowpass::new(400.0, rate), Lowpass::new(400.0, rate)],
                    foam_hp: [Highpass::new(2500.0, rate), Highpass::new(2600.0, rate)],
                    phase: 0.0,
                    period,
                    delay: rng.range(0.05, 0.15),
                }
            }
            Kind::Fire => Voice::Fire {
                brown: Default::default(),
                roar: [Lowpass::new(260.0, rate), Lowpass::new(240.0, rate)],
                flicker: Drift::new(0.4, 6.0, rate, &mut rng),
                crackles: Pings::new(),
                burst: 0,
            },
        };
        Bed { kind, rate, rng, voice, intensity: 0.6 }
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity.clamp(0.0, 1.0);
    }

    /// Fill both channels with the next samples. `left` and `right` must be the same length.
    pub fn fill(&mut self, left: &mut [f32], right: &mut [f32]) {
        debug_assert_eq!(left.len(), right.len());
        let (rate, k) = (self.rate, self.intensity);
        // Each recipe has its own natural level; these bring them all to about -24 dBFS RMS at intensity 0.6, a bed
        // that sits under the music (crates/core/examples/beds.rs measures them).
        let trim = match self.kind {
            Kind::Rain => 1.0,
            Kind::Wind => 0.45,
            Kind::Waves => 0.26,
            Kind::Fire => 0.12,
        };
        let rng = &mut self.rng;
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let (a, b) = match &mut self.voice {
                Voice::Rain { pink, hiss_hp, hiss_lp, drops, heaviness } => {
                    let heavy = 0.6 + 0.4 * heaviness.next(rate, rng);
                    let mut hiss = |i: usize, pink: &mut [Pink; 2], rng: &mut Rng| {
                        let n = pink[i].next(rng.white());
                        hiss_lp[i].next(hiss_hp[i].next(n))
                    };
                    let (hl, hr) = (hiss(0, pink, rng), hiss(1, pink, rng));
                    // Drops: a few hundred a second in a downpour, a few dozen in a drizzle.
                    if event(rng, (30.0 + 500.0 * k) * heavy, rate) {
                        // Mostly small, now and then a big one (a squared random amplitude).
                        let amp = 0.08 * rng.unit().powi(2) + 0.005;
                        let freq = rng.range(1800.0, 6500.0);
                        drops.strike(Ping::strike(freq, rng.range(0.002, 0.006), amp, rng.white(), rate));
                    }
                    let (dl, dr) = drops.next();
                    let level = 0.35 + 0.65 * k;
                    ((hl * 0.55 * heavy + dl) * level, (hr * 0.55 * heavy + dr) * level)
                }
                Voice::Wind { pink, band, body, gust, swirl } => {
                    let g = gust.next(rate, rng);
                    let strength = 0.15 + k * (0.35 + 0.65 * g);
                    let mut out = [0.0; 2];
                    for i in 0..2 {
                        let s = swirl[i].next(rate, rng);
                        // The whistle rises as a gust strengthens.
                        band[i].set(250.0 + 700.0 * g * (0.7 + 0.3 * s) + 120.0 * i as f32, 1.0 + 1.5 * s, rate);
                        let n = pink[i].next(rng.white());
                        out[i] = (band[i].next(n) * 1.6 + body[i].next(n) * 0.9) * strength * (0.75 + 0.25 * s);
                    }
                    (out[0], out[1])
                }
                Voice::Waves { brown, wash, foam_hp, phase, period, delay } => {
                    *phase += 1.0 / (*period * rate);
                    if *phase >= 1.0 {
                        *phase -= 1.0;
                        *period = rng.range(7.0, 13.0);
                    }
                    let mut out = [0.0; 2];
                    for i in 0..2 {
                        // The right channel hears each wave a moment later.
                        let p = (*phase - i as f32 * *delay / *period).rem_euclid(1.0);
                        // Rise over 40 % of the period, break, then a long retreat.
                        let swell = if p < 0.4 {
                            (p / 0.4 * PI / 2.0).sin().powi(2)
                        } else {
                            (1.0 - (p - 0.4) / 0.6).powf(1.6)
                        };
                        wash[i].set(150.0 + 1600.0 * swell, rate);
                        let n = rng.white();
                        let body = wash[i].next(brown[i].next(n)) * (0.25 + 0.75 * swell);
                        let crest = ((p - 0.36) / 0.12).clamp(0.0, 1.0) * (1.0 - ((p - 0.48) / 0.3).clamp(0.0, 1.0));
                        let foam = foam_hp[i].next(n) * crest * 0.18;
                        out[i] = (body * 0.8 + foam) * (0.45 + 0.55 * k);
                    }
                    (out[0], out[1])
                }
                Voice::Fire { brown, roar, flicker, crackles, burst } => {
                    let f = flicker.next(rate, rng);
                    // Crackles come in clusters: a burst raises the rate for a moment.
                    if *burst == 0 && event(rng, 0.4 + 1.5 * k, rate) {
                        *burst = (rng.range(0.05, 0.4) * rate) as u32;
                    }
                    let per_second = if *burst > 0 { 60.0 * (0.4 + k) } else { 4.0 * (0.4 + k) };
                    *burst = burst.saturating_sub(1);
                    if event(rng, per_second, rate) {
                        let pop = rng.unit() < 0.08;
                        let (freq, decay, amp) = if pop {
                            (rng.range(250.0, 700.0), rng.range(0.01, 0.03), rng.range(0.15, 0.35))
                        } else {
                            (rng.range(2500.0, 8000.0), rng.range(0.0006, 0.002), rng.range(0.05, 0.3) * rng.unit())
                        };
                        crackles.strike(Ping::strike(freq, decay, amp, rng.range(-0.6, 0.6), rate));
                    }
                    let (cl, cr) = crackles.next();
                    let size = 0.4 + 0.6 * k;
                    let mut roars = [0.0; 2];
                    for (i, out) in roars.iter_mut().enumerate() {
                        *out = roar[i].next(brown[i].next(rng.white())) * (0.55 + 0.45 * f) * 1.4;
                    }
                    ((roars[0] + cl) * size, (roars[1] + cr) * size)
                }
            };
            *l = a * trim;
            *r = b * trim;
        }
    }
}

/// `seconds` of a bed as a stereo buffer.
pub fn render(kind: Kind, sample_rate: u32, seconds: f32, seed: u64, intensity: f32) -> Audio {
    let n = (seconds * sample_rate as f32).round() as usize;
    let mut bed = Bed::new(kind, sample_rate, seed);
    bed.set_intensity(intensity);
    let (mut l, mut r) = (vec![0.0; n], vec![0.0; n]);
    bed.fill(&mut l, &mut r);
    Audio::new(sample_rate, vec![l, r])
}

/// `seconds` of a bed that loops without a seam, for players that can only repeat a buffer (the browser). The
/// bed runs `fade` seconds longer than the loop, and that extra tail is crossfaded (equal power: the two ends are
/// unrelated noise) into the start, so the last frame leads into the first exactly as the original leads into the
/// tail.
pub fn render_loop(kind: Kind, sample_rate: u32, seconds: f32, fade: f32, seed: u64, intensity: f32) -> Audio {
    let n = (seconds * sample_rate as f32).round() as usize;
    let f = ((fade * sample_rate as f32).round() as usize).min(n);
    let long =
        render(kind, sample_rate, (n + f) as f32 / sample_rate as f32 + 1.0 / sample_rate as f32, seed, intensity);
    let channels = long
        .channels
        .iter()
        .map(|c| {
            let mut out = c[..n].to_vec();
            for (i, s) in out.iter_mut().take(f).enumerate() {
                let x = (i as f32 + 0.5) / f as f32 * PI / 2.0;
                *s = *s * x.sin() + c[n + i] * x.cos();
            }
            out
        })
        .collect();
    Audio::new(sample_rate, channels)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 32_000;

    fn rms(s: &[f32]) -> f32 {
        (s.iter().map(|x| x * x).sum::<f32>() / s.len().max(1) as f32).sqrt()
    }

    /// Energy above `cutoff` as a share of the total (first-difference high-pass is enough to rank beds).
    fn brightness(s: &[f32]) -> f32 {
        let diff: Vec<f32> = s.windows(2).map(|w| w[1] - w[0]).collect();
        rms(&diff) / rms(s).max(1e-9)
    }

    fn correlation(a: &[f32], b: &[f32]) -> f32 {
        let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            ab += f64::from(x * y);
            aa += f64::from(x * x);
            bb += f64::from(y * y);
        }
        (ab / (aa * bb).sqrt().max(1e-12)) as f32
    }

    #[test]
    fn a_loop_wraps_without_a_click_and_keeps_its_level() {
        for kind in Kind::ALL {
            let a = render_loop(kind, RATE, 8.0, 2.0, 11, 0.6);
            assert_eq!(a.len(), 8 * RATE as usize);
            for c in &a.channels {
                // The step across the wrap is like any other step in the bed, not a jump.
                let steps: Vec<f32> = c.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
                let mut sorted = steps.clone();
                sorted.sort_by(f32::total_cmp);
                let p999 = sorted[sorted.len() * 999 / 1000];
                let wrap = (c[0] - c[c.len() - 1]).abs();
                assert!(wrap <= p999.max(1e-3) * 1.5, "{kind:?}: wrap step {wrap} vs 99.9th percentile {p999}");
                // Equal-power crossfade of unrelated noise: the seam region is about as loud as the rest.
                let seam = rms(&c[..RATE as usize]);
                let middle = rms(&c[3 * RATE as usize..7 * RATE as usize]);
                if kind != Kind::Waves {
                    assert!((20.0 * (seam / middle).log10()).abs() < 2.0, "{kind:?}: seam {seam} vs {middle}");
                }
            }
        }
    }

    #[test]
    fn descriptions_pick_beds() {
        assert_eq!(suggest("Steady rain on leaves, distant thunder"), Some(Kind::Rain));
        assert_eq!(suggest("campfire under the stars"), Some(Kind::Fire));
        assert_eq!(suggest("dusk over a quiet harbour"), Some(Kind::Waves));
        assert_eq!(suggest("wind and rain at the seashore"), Some(Kind::Wind));
        assert_eq!(suggest("a medieval tavern"), Some(Kind::Fire));
        assert_eq!(suggest("a new season of research"), None);
        assert_eq!(suggest("warm pads, soft piano"), None);
    }

    #[test]
    fn every_bed_is_finite_bounded_audible_and_stereo() {
        for kind in Kind::ALL {
            let a = render(kind, RATE, 20.0, 7, 0.6);
            let (l, r) = (&a.channels[0], &a.channels[1]);
            assert!(l.iter().chain(r).all(|s| s.is_finite()), "{kind:?} produced NaN or infinity");
            let peak = a.peak();
            assert!(peak < 1.0, "{kind:?} peaks at {peak}");
            let level = rms(l);
            assert!((0.01..0.4).contains(&level), "{kind:?} rms {level}");
            let c = correlation(l, r);
            assert!(c < 0.9, "{kind:?} is nearly mono (correlation {c})");
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_samples_whatever_the_block_size() {
        for kind in Kind::ALL {
            let whole = render(kind, RATE, 3.0, 42, 0.6);
            let mut bed = Bed::new(kind, RATE, 42);
            bed.set_intensity(0.6);
            let n = whole.len();
            let (mut l, mut r) = (vec![0.0; n], vec![0.0; n]);
            let mut at = 0;
            for size in [1, 7, 64, 333, 4096].iter().cycle() {
                if at >= n {
                    break;
                }
                let end = (at + size).min(n);
                let (lb, rb) = (&mut l[at..end], &mut r[at..end]);
                bed.fill(lb, rb);
                at = end;
            }
            assert_eq!(l, whole.channels[0], "{kind:?}");
            assert_ne!(render(kind, RATE, 3.0, 43, 0.6).channels[0], whole.channels[0], "{kind:?}: the seed matters");
        }
    }

    #[test]
    fn rain_is_brighter_than_wind_and_waves() {
        let b = |k| brightness(&render(k, RATE, 10.0, 1, 0.6).channels[0]);
        let (rain, wind, waves) = (b(Kind::Rain), b(Kind::Wind), b(Kind::Waves));
        assert!(rain > wind * 1.5 && rain > waves * 1.5, "rain {rain}, wind {wind}, waves {waves}");
    }

    #[test]
    fn waves_swell_every_7_to_13_seconds() {
        let a = render(Kind::Waves, RATE, 90.0, 3, 0.6);
        // Loudness in quarter-second blocks, then the strongest repeat between 5 and 15 s.
        let block = RATE as usize / 4;
        let env: Vec<f32> = a.channels[0].chunks(block).map(rms).collect();
        let mean = env.iter().sum::<f32>() / env.len() as f32;
        let e: Vec<f32> = env.iter().map(|x| x - mean).collect();
        let best = (20..=60)
            .map(|lag| (lag, e.iter().zip(&e[lag..]).map(|(a, b)| a * b).sum::<f32>()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
            .0;
        let seconds = best as f32 / 4.0;
        assert!((6.5..=13.5).contains(&seconds), "strongest swell period {seconds} s");
        // And the swell is deep: the loudest moments are several times the quietest.
        let mut sorted = env.clone();
        sorted.sort_by(f32::total_cmp);
        let (low, high) = (sorted[sorted.len() / 10], sorted[sorted.len() * 9 / 10]);
        assert!(high > 2.5 * low, "swell depth {high} / {low}");
    }

    #[test]
    fn fire_crackles_and_more_intensity_means_more_rain() {
        // Crackles: sharp peaks well above the roar, at a plausible rate.
        let a = render(Kind::Fire, RATE, 30.0, 5, 0.6);
        let l = &a.channels[0];
        let hp: Vec<f32> = l.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let threshold = 10.0 * rms(&hp);
        let mut clicks = 0;
        let mut last = 0usize;
        for (i, &d) in hp.iter().enumerate() {
            if d > threshold && i > last + RATE as usize / 100 {
                clicks += 1;
                last = i;
            }
        }
        let per_second = clicks as f32 / 30.0;
        assert!((1.0..40.0).contains(&per_second), "{per_second} crackles a second");

        let drops = |k| rms(&render(Kind::Rain, RATE, 10.0, 9, k).channels[0]);
        assert!(drops(1.0) > drops(0.1) * 1.3);
    }
}
