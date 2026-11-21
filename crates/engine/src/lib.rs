//! Live and radio mode as an engine: endless generation, played as it is made, for any audio output.
//!
//! [`Session::start`] spawns two threads and hands back the real-time half:
//! - the **generator** owns the model. It makes one segment at a time, each continuing from the end of the last
//!   (the voice replays the previous segment's final tokens, so the new segment's start reproduces that audio), and
//!   hands it over through a slot that holds one finished segment.
//! - the **control** loop watches the playback clock and schedules each transition a moment ahead: a crossfade into
//!   the next segment inside the overlap, where both sound the same, or, if the next segment is not ready, a crossfade
//!   back to a matching earlier point of the current one (on a bar line for genres with a beat).
//! - [`AudioSide::process`], called from whatever audio callback the caller has (cpal in the CLI, the host in the
//!   plugin), only applies scheduled commands and mixes ([`driftscape_core::mixer`]). It never locks or allocates.
//!
//! The clock is the number of frames `process` has produced. A caller that stops calling it (a DAW with its
//! transport stopped) pauses the session; nothing is lost.
//!
//! [`Session::update`] changes the parameters while it plays: the waiting segment is dropped, the one being made is
//! cancelled, and the next continues from what is audible, short and crossfaded in as soon as it is ready.

use anyhow::{Context, Result, bail};
use driftscape_core::{
    Audio,
    ambience::{self, Bed},
    looping::rejoin_point,
    mixer::{Buffer, Command, Leveler, Mixer},
    preset::{Genre, Station},
    resample::resample,
    tempo,
};
use driftscape_infer::{
    ControlFlow, Request, Tokens, Voice, is_cancelled,
    musicgen::{Backend, MusicGen},
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

pub use driftscape_infer::musicgen::Backend as VoiceBackend;

/// Where models live: `~/Library/Application Support/Driftscape/models` on macOS, `~/.local/share/driftscape/models`
/// on Linux. The CLI and the plugin share it.
pub fn models_dir() -> Result<PathBuf> {
    let base = dirs::data_dir().context("no data directory for this user")?;
    let app = if cfg!(target_os = "macos") { "Driftscape" } else { "driftscape" };
    Ok(base.join(app).join("models"))
}

/// The MusicGen files, checked to be there.
pub fn musicgen_dir() -> Result<PathBuf> {
    let dir = models_dir()?.join("musicgen-small");
    if !dir.join("tokenizer.json").exists() {
        bail!("MusicGen is not in {}; download Xenova/musicgen-small (q4 ONNX files) there", dir.display());
    }
    Ok(dir)
}

/// What the listener can change while it plays. Read by the generator before each segment.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    pub genre: Genre,
    pub prompt: String,
    /// Added to the genre's guidance: how closely to follow the prompt.
    pub focus: f32,
    /// Added to the genre's temperature: how much the music wanders.
    pub variety: f32,
}

impl Params {
    pub fn new(genre: Genre, prompt: impl Into<String>) -> Self {
        Self { genre, prompt: prompt.into(), focus: 0.0, variety: 0.0 }
    }
}

/// The ambience bed under the music.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BedChoice {
    pub kind: ambience::Kind,
    /// Gain in dB; at 0 the bed sits a few dB under the music.
    pub db: f32,
    /// 0..=1: how much rain, wind, sea or fire.
    pub intensity: f32,
}

impl BedChoice {
    /// The bed a genre puts under the listener's words ([`Genre::bed_for`]), at the genre's level.
    pub fn for_params(params: &Params) -> Option<BedChoice> {
        let kind = params.genre.bed_for(&params.prompt)?;
        Some(BedChoice { kind, db: params.genre.bed_db.unwrap_or(0.0), intensity: 0.6 })
    }
}

pub struct Config {
    pub params: Params,
    /// The ambience bed to start with.
    pub bed: Option<BedChoice>,
    /// A station's prompts replace `params.prompt`, one after another.
    pub station: Option<Station>,
    pub backend: Backend,
    /// The output's sample rate; segments are resampled to it.
    pub rate: u32,
}

/// What happened, for the caller to show. Sent from the control and generator threads, never from `process`.
#[derive(Debug, Clone)]
pub enum Event {
    /// The model is loaded and the first segment is being made.
    Loaded,
    /// A new segment started playing.
    Playing {
        label: String,
    },
    /// The current segment repeats while the next is made.
    Repeating,
    /// `process` stopped being called while music played (a device change, or a stopped transport).
    Paused,
    Resumed,
    /// The generator stopped with an error; the session ends.
    Failed(String),
    /// Diagnostics, sent when `DRIFTSCAPE_TRACE` is set.
    Trace(String),
}

/// State shared by the generator, the control loop and [`Session::update`].
struct Shared {
    params: Params,
    /// The finished segment waiting to play next.
    ready: Option<Segment>,
    /// Tokens of the segment playing now. A generation that a tweak cancels restarts from here, so the music still
    /// continues from what is audible.
    playing: Option<Tokens>,
}

/// A finished segment, at the output rate.
struct Segment {
    buffer: Buffer,
    tokens: Tokens,
    /// Frames at the start that replay the previous segment's end.
    overlap: usize,
    /// Crossfade into this segment, and out of it when it repeats.
    crossfade: usize,
    /// Where to jump back to if this segment has to repeat: its last `crossfade` frames sound like the ones from here.
    rejoin: usize,
    label: String,
    /// The first segment after a tweak: it comes in as soon as it is ready instead of at the current one's end.
    urgent: bool,
}

/// Mixer state the control loop tracks.
struct Playing {
    id: u64,
    segment: Arc<Segment>,
    /// The time at which buffer frame `offset` plays.
    start: u64,
    offset: usize,
}

impl Playing {
    /// When this voice's buffer runs out.
    fn end(&self) -> u64 {
        self.start + (self.segment.buffer.len() - self.offset) as u64
    }
}

/// New audio in the first segment after a tweak.
const URGENT_SECONDS: f32 = 6.0;
/// Crossfade for a change the listener asked for, which doesn't wait for matching material.
const CHANGE_FADE_SECONDS: f32 = 3.0;
/// Typical loudness for a segment (about -20 dBFS RMS), and limits on the gain used to get there.
pub const TARGET_RMS: f32 = 0.1;
const MAX_GAIN: f32 = 8.0;
const PEAK_LIMIT: f32 = 0.95;

/// A running session. Dropping it stops it.
pub struct Session {
    shared: Arc<Mutex<Shared>>,
    bed: Mutex<(rtrb::Producer<Option<BedChoice>>, Option<BedChoice>)>,
    version: Arc<AtomicU64>,
    quit: Arc<AtomicBool>,
    events: mpsc::Receiver<Event>,
    threads: Vec<JoinHandle<()>>,
}

/// The real-time half: call [`AudioSide::process`] from the audio callback.
pub struct AudioSide {
    commands: rtrb::Consumer<Command>,
    finished: rtrb::Producer<Buffer>,
    clock: Arc<AtomicU64>,
    mixer: Mixer,
    leveler: Leveler,
    beds: BedMixer,
    bed_changes: rtrb::Consumer<Option<BedChoice>>,
}

impl AudioSide {
    /// Fill `left` and `right` (the same length) with the next stereo frames: the music in the middle, the bed around
    /// it. Advances the clock by the block's length. Real-time safe: no locks, no allocation, no freeing (finished
    /// buffers go back to the control thread).
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        while let Ok(command) = self.commands.pop() {
            self.mixer.apply(command);
        }
        while let Ok(change) = self.bed_changes.pop() {
            self.beds.change(change);
        }
        self.mixer.render(left);
        self.leveler.process(left);
        right.copy_from_slice(left);
        self.beds.add(left, right);
        for buffer in self.mixer.take_finished() {
            let _ = self.finished.push(buffer);
        }
        self.clock.store(self.mixer.time(), Ordering::Release);
    }
}

/// The bed's scratch size; [`AudioSide::process`] works through longer blocks in pieces of this.
pub const MAX_BLOCK: usize = 2048;

/// Up to two beds: the current one, and the one it replaced, fading out. Beds hold no heap memory, so making and
/// dropping them in the audio callback is real-time safe.
struct BedMixer {
    rate: u32,
    seed: u64,
    slots: [Option<BedSlot>; 2],
    scratch: [Box<[f32; MAX_BLOCK]>; 2],
}

struct BedSlot {
    bed: Bed,
    gain: f32,
    target: f32,
    /// Per-sample smoothing towards `target` (about half a second).
    coef: f32,
}

impl BedMixer {
    fn new(rate: u32, seed: u64, initial: Option<BedChoice>) -> Self {
        let scratch = [Box::new([0.0; MAX_BLOCK]), Box::new([0.0; MAX_BLOCK])];
        let mut m = BedMixer { rate, seed, slots: [None, None], scratch };
        m.change(initial);
        m
    }

    fn change(&mut self, choice: Option<BedChoice>) {
        let coef = 1.0 - (-1.0 / (0.5 * self.rate as f32)).exp();
        match (self.slots[0].as_mut(), choice) {
            // Same kind: just a new level and intensity.
            (Some(slot), Some(c)) if slot.bed.kind() == c.kind => {
                slot.target = 10f32.powf(c.db / 20.0);
                slot.bed.set_intensity(c.intensity);
            }
            _ => {
                // Fade the current one out (replacing any older one still fading) and bring the new one in.
                let mut old = self.slots[0].take();
                if let Some(o) = old.as_mut() {
                    o.target = 0.0;
                }
                self.slots[1] = old;
                if let Some(c) = choice {
                    self.seed = self.seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    let mut bed = Bed::new(c.kind, self.rate, self.seed);
                    bed.set_intensity(c.intensity);
                    self.slots[0] = Some(BedSlot { bed, gain: 0.0, target: 10f32.powf(c.db / 20.0), coef });
                }
            }
        }
    }

    fn add(&mut self, left: &mut [f32], right: &mut [f32]) {
        for (l_block, r_block) in left.chunks_mut(MAX_BLOCK).zip(right.chunks_mut(MAX_BLOCK)) {
            let n = l_block.len();
            for slot in self.slots.iter_mut().flatten() {
                let [bl, br] = &mut self.scratch;
                let (bl, br) = (&mut bl[..n], &mut br[..n]);
                slot.bed.fill(bl, br);
                for i in 0..n {
                    slot.gain += slot.coef * (slot.target - slot.gain);
                    l_block[i] += bl[i] * slot.gain;
                    r_block[i] += br[i] * slot.gain;
                }
            }
        }
        if self.slots[1].as_ref().is_some_and(|s| s.target == 0.0 && s.gain < 1e-3) {
            self.slots[1] = None;
        }
    }
}

/// Stops a session from another thread (a Ctrl-C handler).
#[derive(Clone)]
pub struct Stopper(Arc<AtomicBool>);

impl Stopper {
    pub fn stop(&self) {
        self.0.store(true, Ordering::Release);
    }
    /// Whether a stop was asked for; true if it already was.
    pub fn swap(&self) -> bool {
        self.0.swap(true, Ordering::SeqCst)
    }
}

impl Session {
    /// Start generating. Music begins once the first segment is ready (a few seconds on a recent Mac's GPU).
    pub fn start(config: Config) -> Result<(Session, AudioSide)> {
        let rate = config.rate;
        let shared = Arc::new(Mutex::new(Shared { params: config.params, ready: None, playing: None }));
        let version = Arc::new(AtomicU64::new(0));
        let quit = Arc::new(AtomicBool::new(false));
        let clock = Arc::new(AtomicU64::new(0));
        let (event_tx, events) = mpsc::channel();
        let dir = musicgen_dir()?;

        let (command_tx, commands) = rtrb::RingBuffer::<Command>::new(64);
        let (finished, finished_rx) = rtrb::RingBuffer::<Buffer>::new(64);
        let (bed_tx, bed_changes) = rtrb::RingBuffer::<Option<BedChoice>>::new(16);
        let seed =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);

        let generator = {
            let (shared, version, quit, events) = (shared.clone(), version.clone(), quit.clone(), event_tx.clone());
            let (station, backend) = (config.station, config.backend);
            thread::Builder::new()
                .name("driftscape generator".into())
                .spawn(move || generate_loop(dir, backend, rate, shared, version, station, quit, events))?
        };
        let control = {
            let (shared, quit, clock) = (shared.clone(), quit.clone(), clock.clone());
            thread::Builder::new()
                .name("driftscape control".into())
                .spawn(move || control_loop(rate, shared, quit, clock, command_tx, finished_rx, event_tx))?
        };

        let audio = AudioSide {
            commands,
            finished,
            clock,
            mixer: Mixer::new(),
            leveler: Leveler::new(rate, TARGET_RMS, 3.0),
            beds: BedMixer::new(rate, seed, config.bed),
            bed_changes,
        };
        let bed = Mutex::new((bed_tx, config.bed));
        Ok((Session { shared, bed, version, quit, events, threads: vec![generator, control] }, audio))
    }

    pub fn params(&self) -> Params {
        self.shared.lock().unwrap().params.clone()
    }

    /// Change the parameters. If they changed, the next part is made with them from what is playing now and comes in
    /// as soon as it is ready. Returns whether anything changed.
    pub fn update(&self, change: impl FnOnce(&mut Params)) -> bool {
        let mut s = self.shared.lock().unwrap();
        let before = s.params.clone();
        change(&mut s.params);
        if s.params == before {
            return false;
        }
        s.ready = None;
        self.version.fetch_add(1, Ordering::AcqRel);
        true
    }

    /// Change the bed under the music (or remove it): it crossfades over about a second. The music isn't touched.
    pub fn set_bed(&self, choice: Option<BedChoice>) {
        let mut bed = self.bed.lock().unwrap();
        if bed.1 != choice && bed.0.push(choice).is_ok() {
            bed.1 = choice;
        }
    }

    pub fn bed(&self) -> Option<BedChoice> {
        self.bed.lock().unwrap().1
    }

    /// The next event, if one is waiting.
    pub fn try_event(&self) -> Option<Event> {
        self.events.try_recv().ok()
    }

    /// Wait up to `timeout` for the next event.
    pub fn event_timeout(&self, timeout: Duration) -> Option<Event> {
        self.events.recv_timeout(timeout).ok()
    }

    pub fn is_running(&self) -> bool {
        !self.quit.load(Ordering::Acquire)
    }

    pub fn stopper(&self) -> Stopper {
        Stopper(self.quit.clone())
    }

    /// Stop and wait for the threads to finish (at most the current model step).
    pub fn stop(&mut self) {
        self.quit.store(true, Ordering::Release);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Take the waiting segment, if any, and mark it as the one playing. One lock, so a tweak can't slip in between.
fn take_ready(shared: &Mutex<Shared>) -> Option<Segment> {
    let mut s = shared.lock().unwrap();
    let segment = s.ready.take()?;
    s.playing = Some(segment.tokens.clone());
    Some(segment)
}

fn push(tx: &mut rtrb::Producer<Command>, command: Command, events: &mpsc::Sender<Event>) {
    if tx.push(command).is_err() {
        let _ = events.send(Event::Trace("audio command queue full".into()));
    }
}

fn control_loop(
    rate: u32,
    shared: Arc<Mutex<Shared>>,
    quit: Arc<AtomicBool>,
    clock: Arc<AtomicU64>,
    mut command_tx: rtrb::Producer<Command>,
    mut finished_rx: rtrb::Consumer<Buffer>,
    events: mpsc::Sender<Event>,
) {
    const LOOKAHEAD: u64 = 4800; // schedule transitions ~0.1 s ahead of time at 48 kHz
    const STALL: Duration = Duration::from_secs(2);
    let lookahead = LOOKAHEAD * u64::from(rate) / 48_000;
    let mut playing: Option<Playing> = None;
    let mut next_id = 1u64;
    let trace = std::env::var_os("DRIFTSCAPE_TRACE").is_some();
    let (mut last_clock, mut stalled_since, mut paused) = (0u64, None::<Instant>, false);
    let mut last_trace = u64::MAX;
    let send = |e: Event| {
        let _ = events.send(e);
    };

    while !quit.load(Ordering::Acquire) {
        while let Ok(buffer) = finished_rx.pop() {
            drop(buffer); // freed here, not in the audio callback
        }
        let now = clock.load(Ordering::Acquire);
        // The clock only moves while `process` is called. If it stops while music plays, say so.
        if now != last_clock {
            stalled_since = None;
            if paused {
                paused = false;
                send(Event::Resumed);
            }
            last_clock = now;
        } else if playing.is_some() {
            let since = *stalled_since.get_or_insert_with(Instant::now);
            if since.elapsed() > STALL && !paused {
                paused = true;
                send(Event::Paused);
            }
        }
        if trace && now / u64::from(rate) != last_trace {
            last_trace = now / u64::from(rate);
            let ready = shared.lock().unwrap().ready.as_ref().map(|r| r.urgent);
            send(Event::Trace(format!(
                "clock {:.1} s · playing ends {:?} · ready {:?}",
                now as f64 / f64::from(rate),
                playing.as_ref().map(|p| format!("{:.1}", p.end() as f64 / f64::from(rate))),
                ready
            )));
        }
        match &playing {
            None => {
                // Wait for the first segment, then start it with a short fade-in.
                if let Some(segment) = take_ready(&shared) {
                    let segment = Arc::new(segment);
                    let at = now + lookahead;
                    let id = next_id;
                    next_id += 1;
                    let fade_in = rate as usize / 2;
                    push(
                        &mut command_tx,
                        Command::Play { id, buffer: segment.buffer.clone(), offset: 0, at, fade_in },
                        &events,
                    );
                    send(Event::Playing { label: segment.label.clone() });
                    playing = Some(Playing { id, segment, start: at, offset: 0 });
                }
            }
            Some(current) => {
                // A change the listener asked for comes in now, with a longer crossfade, once the current segment's
                // own fade-in is over.
                let urgent_ready = shared.lock().unwrap().ready.as_ref().is_some_and(|r| r.urgent);
                let change_fade = (CHANGE_FADE_SECONDS * rate as f32) as usize;
                let at_now = now + lookahead;
                if urgent_ready
                    && at_now >= current.start + current.segment.crossfade as u64
                    && at_now + (change_fade as u64) < current.end()
                    && let Some(segment) = take_ready(&shared)
                {
                    let id = next_id;
                    next_id += 1;
                    let offset = segment.overlap.saturating_sub(change_fade);
                    send(Event::Playing { label: segment.label.clone() });
                    push(&mut command_tx, Command::FadeOut { id: current.id, at: at_now, fade: change_fade }, &events);
                    push(
                        &mut command_tx,
                        Command::Play { id, buffer: segment.buffer.clone(), offset, at: at_now, fade_in: change_fade },
                        &events,
                    );
                    playing = Some(Playing { id, segment: Arc::new(segment), start: at_now, offset });
                    continue;
                }
                let fade = current.segment.crossfade;
                let at = current.end() - fade as u64;
                if now + lookahead >= at {
                    let id = next_id;
                    next_id += 1;
                    let next = match take_ready(&shared) {
                        Some(segment) => {
                            // The new segment's frame `overlap - fade` sounds like the current one's `len - fade`.
                            let offset = segment.overlap.saturating_sub(fade);
                            send(Event::Playing { label: segment.label.clone() });
                            Playing { id, segment: Arc::new(segment), start: at, offset }
                        }
                        None => {
                            send(Event::Repeating);
                            Playing { id, segment: current.segment.clone(), start: at, offset: current.segment.rejoin }
                        }
                    };
                    push(&mut command_tx, Command::FadeOut { id: current.id, at, fade }, &events);
                    push(
                        &mut command_tx,
                        Command::Play {
                            id,
                            buffer: next.segment.buffer.clone(),
                            offset: next.offset,
                            at,
                            fade_in: fade,
                        },
                        &events,
                    );
                    playing = Some(next);
                }
            }
        }
        thread::sleep(Duration::from_millis(10));
    }
}

/// Make segments until told to stop.
///
/// Each segment continues from the previous one. A tweak (a change of `version`) cancels the segment being made and
/// discards a finished one that has not started; the next one then continues from the segment playing now.
#[allow(clippy::too_many_arguments)]
fn generate_loop(
    dir: PathBuf,
    backend: Backend,
    rate: u32,
    shared: Arc<Mutex<Shared>>,
    version: Arc<AtomicU64>,
    station: Option<Station>,
    quit: Arc<AtomicBool>,
    events: mpsc::Sender<Event>,
) {
    let result = (|| -> Result<()> {
        let env = ort::environment::init().with_name("driftscape").with_telemetry(false).build()?;
        let mut voice = MusicGen::load(&env, &dir, backend)?;
        let _ = events.send(Event::Loaded);
        let mut previous: Option<Tokens> = None;
        let mut gain = None::<f32>;
        let mut seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos() as u64;
        let mut number = 0u64;
        let mut placed_version = 0u64; // the tweak version of the last segment handed over
        'segments: while !quit.load(Ordering::Acquire) {
            let v = version.load(Ordering::Acquire);
            let p = shared.lock().unwrap().params.clone();
            let listener = match &station {
                Some(s) => s.prompt_for(number),
                None => &p.prompt,
            };
            let g = &p.genre;
            let continue_from = previous.as_ref().map(|t| t.tail(g.overlap_seconds));
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // Right at the start and after a tweak, a short segment gets music (or the change) to the listener
            // sooner. Otherwise full length: every segment first replays the overlap, which costs time without adding
            // new audio, so long segments are what keep generation ahead of playback.
            let urgent = v != placed_version || number == 0;
            let seconds = if urgent { g.segment_seconds.min(URGENT_SECONDS) } else { g.segment_seconds };
            let request = Request {
                guidance: (g.guidance + p.focus).clamp(1.0, 8.0),
                temperature: (g.temperature + p.variety).clamp(0.5, 1.6),
                continue_from,
                ..Request::new(g.prompt(listener), seconds, seed)
            };
            let started = Instant::now();
            let stale = || quit.load(Ordering::Acquire) || version.load(Ordering::Acquire) != v;
            let clip = match voice.generate(&request, &mut |_| {
                if stale() { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
            }) {
                Ok(clip) => clip,
                Err(e) if is_cancelled(&e) => {
                    previous = shared.lock().unwrap().playing.clone();
                    continue;
                }
                Err(e) => return Err(e),
            };
            let took = started.elapsed().as_secs_f32();

            let audio = resample(&clip.audio, rate);
            let samples = &audio.channels[0];
            // Loudness: aim for a steady level, change it gradually between segments, never clip.
            let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt();
            let peak = audio.peak().max(1e-6);
            let wanted = (TARGET_RMS / rms.max(1e-6)).min(MAX_GAIN);
            let g_now = gain.map_or(wanted, |prev| 0.5 * prev + 0.5 * wanted).min(PEAK_LIMIT / peak);
            gain = Some(g_now);
            let buffer: Buffer = samples.iter().map(|s| s * g_now).collect::<Vec<_>>().into();

            let overlap = (clip.overlap_seconds * rate as f32) as usize;
            let crossfade = ((g.crossfade_seconds * rate as f32) as usize).min(overlap.max(rate as usize));
            let rejoin = rejoin_for(&audio, g, rate, overlap, crossfade);
            // For genres with a beat, report the tempo the new part actually plays at.
            let pulse = match g.bpm {
                Some(_) => match tempo::estimate(&samples[overlap.min(samples.len())..], rate, tempo::Range::default())
                {
                    Some(t) => format!(" · {:.1} BPM", t.bpm),
                    None => " · no pulse".to_owned(),
                },
                None => String::new(),
            };
            let label = format!(
                "segment {} · {} · {} · {:.0} s in {:.1} s ({:.1}× real time){pulse}",
                number + 1,
                g.name,
                truncate(listener_or_default(listener, g), 48),
                seconds,
                took,
                seconds / took
            );
            let mut segment = Some(Segment { buffer, tokens: clip.tokens, overlap, crossfade, rejoin, label, urgent });

            // Wait for the slot to free up; if a tweak arrives meanwhile, this segment is stale.
            loop {
                if quit.load(Ordering::Acquire) {
                    break 'segments;
                }
                {
                    let mut s = shared.lock().unwrap();
                    if version.load(Ordering::Acquire) != v {
                        previous = s.playing.clone();
                        continue 'segments;
                    }
                    if s.ready.is_none() {
                        let seg = segment.take().expect("placed once");
                        previous = Some(seg.tokens.clone());
                        s.ready = Some(seg);
                        placed_version = v;
                        break;
                    }
                }
                thread::sleep(Duration::from_millis(20));
            }
            number += 1;
        }
        Ok(())
    })();
    if let Err(e) = result {
        let _ = events.send(Event::Failed(format!("{e:#}")));
        quit.store(true, Ordering::Release);
    }
}

/// Where a segment should jump back to if it has to repeat: a point whose sound matches its end, at least a few
/// seconds back so repeats are not short, and on a bar line when the genre has a beat.
fn rejoin_for(audio: &Audio, genre: &Genre, rate: u32, overlap: usize, crossfade: usize) -> usize {
    let samples = &audio.channels[0];
    let from = samples.len() - crossfade;
    // Repeats at least 8 s long where the segment allows it, so a repeat is a phrase returning, not a stutter.
    let min_loop = (8 * rate as usize).min(from.saturating_sub(rate as usize));
    let latest = from.saturating_sub(min_loop);
    let bar = genre.bpm.map(|requested| {
        // Prefer the tempo the model actually played, if it is close to the one asked for.
        let played = tempo::estimate(samples, rate, tempo::Range { min_bpm: 60.0, max_bpm: 180.0 })
            .map(|t| t.bpm)
            .filter(|b| (b - f64::from(requested)).abs() < 6.0)
            .unwrap_or(f64::from(requested));
        (4.0 * 60.0 / played * f64::from(rate)).round() as usize
    });
    rejoin_point(samples, from, crossfade.max(1), overlap.min(latest), latest, bar)
        .or_else(|| rejoin_point(samples, from, crossfade.max(1), 0, latest, None))
        .map_or(overlap, |(s, _)| s)
}

/// The listener's words, or the genre's default prompt when there are none.
pub fn listener_or_default<'a>(listener: &'a str, genre: &'a Genre) -> &'a str {
    if listener.trim().is_empty() { &genre.default_prompt } else { listener }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_owned() } else { format!("{}…", s.chars().take(n - 1).collect::<String>()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use driftscape_core::ambience::Kind;

    fn choice(kind: Kind) -> Option<BedChoice> {
        Some(BedChoice { kind, db: 0.0, intensity: 0.6 })
    }

    #[test]
    fn beds_crossfade_without_clicks_and_fade_out_when_removed() {
        let rate = 32_000;
        let mut m = BedMixer::new(rate, 1, choice(Kind::Rain));
        let block = 300; // not a divisor of MAX_BLOCK, on purpose
        let mut out = Vec::new();
        let run = |m: &mut BedMixer, out: &mut Vec<f32>, seconds: f32| {
            for _ in 0..(seconds * rate as f32) as usize / block {
                let (mut l, mut r) = (vec![0.0; block], vec![0.0; block]);
                m.add(&mut l, &mut r);
                out.extend_from_slice(&l);
            }
        };
        run(&mut m, &mut out, 2.0);
        let switch = out.len();
        m.change(choice(Kind::Wind));
        run(&mut m, &mut out, 4.0);
        assert!(m.slots[1].is_none(), "the old bed is dropped once it has faded out");
        // The bed's own largest step, before the change, bounds the steps around it: a click would exceed it.
        let step = |s: &[f32]| s.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        let before = step(&out[rate as usize..switch]);
        let around = step(&out[switch..switch + rate as usize / 10]);
        assert!(around < before * 1.5, "switching steps {around} vs {before} before");

        m.change(None);
        let quiet_from = out.len() + 3 * rate as usize;
        run(&mut m, &mut out, 4.0);
        let tail = &out[quiet_from..];
        assert!(tail.iter().all(|s| s.abs() < 1e-3), "silent once removed");
    }
}
