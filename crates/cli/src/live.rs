//! Live and radio mode: endless generation, played as it is made.
//!
//! Four threads:
//! - the **generator** owns the model. It makes one segment at a time, each continuing from the end of the last
//!   (the voice replays the previous segment's final tokens, so the new segment's start reproduces that audio), and
//!   hands it over through a channel that holds one finished segment.
//! - the **control** loop (the main thread) watches the playback clock and schedules each transition a moment ahead:
//!   a crossfade into the next segment inside the overlap, where both sound the same, or, if the next segment is not
//!   ready, a crossfade back to a matching earlier point of the current one (on a bar line for genres with a beat).
//! - the **audio callback** only applies scheduled commands and mixes ([`driftscape_core::mixer`]).
//! - the **recorder** streams everything that is played to a WAV file.
//!
//! Tweaks typed while it plays change the next segment.

use crate::{generate::models_dir, recorder::WavRecorder};
use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use driftscape_core::{
    Audio,
    looping::rejoin_point,
    mixer::{Buffer, Command, Leveler, Mixer},
    preset::{Genre, Station, builtin_genres},
    resample::resample,
    tempo,
};
use driftscape_infer::{
    ControlFlow, Request, Tokens, Voice, is_cancelled,
    musicgen::{Backend, MusicGen},
};
use std::{
    io::BufRead,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

/// What drives the session.
pub enum Source {
    Live { genre: Genre, prompt: String },
    Radio { station: Station, genre: Genre },
}

pub struct Options {
    pub source: Source,
    pub backend: Backend,
    /// Where to record; `None` disables recording.
    pub record: Option<PathBuf>,
    /// Stop after this many minutes.
    pub stop_after: Option<f64>,
}

/// What the listener can change while it plays. Read by the generator before each segment.
#[derive(Debug, Clone)]
struct Params {
    genre: Genre,
    prompt: String,
    /// Added to the genre's guidance: how closely to follow the prompt.
    focus: f32,
    /// Added to the genre's temperature: how much the music wanders.
    variety: f32,
}

/// State shared by the generator, the control loop and the input thread.
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
const TARGET_RMS: f32 = 0.1;
const MAX_GAIN: f32 = 8.0;
const PEAK_LIMIT: f32 = 0.95;

pub fn run(options: Options) -> Result<()> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or_else(|| anyhow!("no audio output device"))?;
    let config = device.default_output_config().context("querying the output device")?;
    if config.sample_format() != cpal::SampleFormat::F32 {
        bail!("output device wants {:?} samples; only f32 is supported so far", config.sample_format());
    }
    // Latency doesn't matter here and the model keeps the machine busy, so ask for a roomy buffer.
    let buffer_size = match config.buffer_size() {
        cpal::SupportedBufferSize::Range { min, max } => cpal::BufferSize::Fixed(2048.clamp(*min, *max)),
        cpal::SupportedBufferSize::Unknown => cpal::BufferSize::Default,
    };
    let mut config: cpal::StreamConfig = config.into();
    config.buffer_size = buffer_size;
    let rate = config.sample_rate;
    let channels = usize::from(config.channels);

    let (genre, prompt, station) = match options.source {
        Source::Live { genre, prompt } => (genre, prompt, None),
        Source::Radio { station, genre } => (genre, String::new(), Some(station)),
    };
    let shared = Arc::new(Mutex::new(Shared {
        params: Params { genre, prompt, focus: 0.0, variety: 0.0 },
        ready: None,
        playing: None,
    }));
    // Bumped by every tweak. The generator checks it every step and starts over when it changes.
    let version = Arc::new(AtomicU64::new(0));
    let quit = Arc::new(AtomicBool::new(false));
    {
        let quit = quit.clone();
        // The first Ctrl-C stops cleanly (and saves the recording); a second one exits at once, whatever is stuck.
        ctrlc::set_handler(move || {
            if quit.swap(true, Ordering::SeqCst) {
                eprintln!("\nstopped");
                std::process::exit(130);
            }
        })
        .context("installing the Ctrl-C handler")?;
    }

    let clock = Arc::new(AtomicU64::new(0));
    let generator = {
        let (shared, version, quit, station) = (shared.clone(), version.clone(), quit.clone(), station.clone());
        thread::Builder::new()
            .name("generator".into())
            .spawn(move || generate_loop(options.backend, rate, shared, version, station, quit))?
    };

    // Control → audio: commands; audio → control: finished buffers to drop; audio → recorder: played samples.
    let (mut command_tx, mut command_rx) = rtrb::RingBuffer::<Command>::new(64);
    let (mut finished_tx, mut finished_rx) = rtrb::RingBuffer::<Buffer>::new(64);
    let (mut record_tx, mut record_rx) = rtrb::RingBuffer::<f32>::new(rate as usize * 4);
    let callback_clock = clock.clone();
    let mut mixer = Mixer::new();
    let mut leveler = Leveler::new(rate, TARGET_RMS, 3.0);
    let mut scratch = vec![0.0f32; 8192];
    let recording = options.record.is_some();

    let stream = device
        .build_output_stream(
            config,
            move |out: &mut [f32], _| {
                while let Ok(command) = command_rx.pop() {
                    mixer.apply(command);
                }
                for frames in out.chunks_mut(scratch.len() * channels) {
                    let n = frames.len() / channels;
                    let mono = &mut scratch[..n];
                    mixer.render(mono);
                    leveler.process(mono);
                    for (frame, &s) in frames.chunks_mut(channels).zip(mono.iter()) {
                        frame.fill(s);
                    }
                    if recording {
                        for &s in mono.iter() {
                            let _ = record_tx.push(s);
                        }
                    }
                }
                for buffer in mixer.take_finished() {
                    let _ = finished_tx.push(buffer);
                }
                callback_clock.store(mixer.time(), Ordering::Release);
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .context("opening the audio stream")?;
    stream.play().context("starting playback")?;

    let recorder = match &options.record {
        Some(path) => {
            let mut wav = WavRecorder::create(path, rate)?;
            let quit = quit.clone();
            Some(thread::Builder::new().name("recorder".into()).spawn(move || -> Result<WavRecorder> {
                let mut chunk = Vec::with_capacity(rate as usize);
                let mut started = false; // skip the silence while the first segment is generated
                loop {
                    let done = quit.load(Ordering::Acquire);
                    while let Ok(s) = record_rx.pop() {
                        started |= s.abs() > 1e-4;
                        if started {
                            chunk.push(s);
                        }
                    }
                    wav.write(&chunk)?;
                    chunk.clear();
                    if done {
                        return Ok(wav);
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            })?)
        }
        None => None,
    };

    spawn_input(shared.clone(), version.clone(), quit.clone(), station.is_some());
    match &station {
        Some(s) => eprintln!("📻 {} · {}", s.name, s.description),
        None => {
            let p = &shared.lock().unwrap().params;
            eprintln!(
                "live · {} · {}",
                p.genre.name,
                if p.prompt.is_empty() { &p.genre.default_prompt } else { &p.prompt }
            );
        }
    }
    eprintln!("type a new prompt and press Enter to steer; :help lists commands; Ctrl-C stops");
    if let Some(path) = &options.record {
        eprintln!("recording to {}", path.display());
    }
    eprintln!("generating the first segment…");

    // The control loop.
    const LOOKAHEAD: u64 = 4800; // schedule transitions ~0.1 s ahead of time at 48 kHz
    let lookahead = LOOKAHEAD * u64::from(rate) / 48_000;
    let mut playing: Option<Playing> = None;
    let mut next_id = 1u64;
    let trace = std::env::var_os("DRIFTSCAPE_TRACE").is_some();
    const STALL: Duration = Duration::from_secs(2);
    let (mut last_clock, mut stalled_since, mut stall_reported) = (0u64, None::<std::time::Instant>, false);
    let mut last_trace = u64::MAX;
    let deadline = options.stop_after.map(|m| std::time::Instant::now() + Duration::from_secs_f64(m * 60.0));
    while !quit.load(Ordering::Acquire) {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            quit.store(true, Ordering::Release);
            break;
        }
        while let Ok(buffer) = finished_rx.pop() {
            drop(buffer); // freed here, not in the audio callback
        }
        let now = clock.load(Ordering::Acquire);
        // The audio callback advances the clock. If it stops (the output device went away or changed), say so
        // instead of waiting silently.
        if now != last_clock {
            if stalled_since.take().is_some_and(|t| t.elapsed() > STALL) {
                eprintln!("  audio output resumed");
            }
            last_clock = now;
        } else if playing.is_some() {
            let since = *stalled_since.get_or_insert_with(std::time::Instant::now);
            if since.elapsed() > STALL && !stall_reported {
                stall_reported = true;
                eprintln!(
                    "  the audio output stopped (did the output device change?); Ctrl-C to stop, or reconnect the device"
                );
            }
        }
        if stalled_since.is_none() {
            stall_reported = false;
        }
        if trace && now / u64::from(rate) != last_trace {
            last_trace = now / u64::from(rate);
            let ready = shared.lock().unwrap().ready.as_ref().map(|r| r.urgent);
            eprintln!(
                "[trace] clock {:.1} s · playing ends {:?} · ready {:?}",
                now as f64 / f64::from(rate),
                playing.as_ref().map(|p| format!("{:.1}", p.end() as f64 / f64::from(rate))),
                ready
            );
        }
        match &playing {
            None => {
                // Wait for the first segment, then start it with a short fade-in.
                if let Some(segment) = take_ready(&shared) {
                    let segment = Arc::new(segment);
                    let at = now + lookahead;
                    let id = next_id;
                    next_id += 1;
                    push(
                        &mut command_tx,
                        Command::Play { id, buffer: segment.buffer.clone(), offset: 0, at, fade_in: rate as usize / 2 },
                    );
                    eprintln!("▶ {}", segment.label);
                    playing = Some(Playing { id, segment, start: at, offset: 0 });
                }
            }
            Some(current) => {
                // A change the listener asked for comes in now, with a longer crossfade, once the current
                // segment's own fade-in is over.
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
                    eprintln!("▶ {}", segment.label);
                    push(&mut command_tx, Command::FadeOut { id: current.id, at: at_now, fade: change_fade });
                    push(
                        &mut command_tx,
                        Command::Play { id, buffer: segment.buffer.clone(), offset, at: at_now, fade_in: change_fade },
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
                            eprintln!("▶ {}", segment.label);
                            Playing { id, segment: Arc::new(segment), start: at, offset }
                        }
                        None => {
                            eprintln!("↺ repeating while the next segment is generated");
                            Playing { id, segment: current.segment.clone(), start: at, offset: current.segment.rejoin }
                        }
                    };
                    push(&mut command_tx, Command::FadeOut { id: current.id, at, fade });
                    push(
                        &mut command_tx,
                        Command::Play {
                            id,
                            buffer: next.segment.buffer.clone(),
                            offset: next.offset,
                            at,
                            fade_in: fade,
                        },
                    );
                    playing = Some(next);
                }
            }
        }
        thread::sleep(Duration::from_millis(10));
    }

    eprintln!("\nstopping…");
    drop(stream);
    if let Some(recorder) = recorder {
        let wav = recorder.join().map_err(|_| anyhow!("recorder thread panicked"))??;
        let seconds = wav.seconds(rate);
        let path = wav.finish()?;
        eprintln!("saved {:.0} s of this session to {}", seconds, path.display());
    }
    let _ = generator.join();
    Ok(())
}

/// Take the waiting segment, if any, and mark it as the one playing. One lock, so a tweak can't slip in between.
fn take_ready(shared: &Mutex<Shared>) -> Option<Segment> {
    let mut s = shared.lock().unwrap();
    let segment = s.ready.take()?;
    s.playing = Some(segment.tokens.clone());
    Some(segment)
}

fn push(tx: &mut rtrb::Producer<Command>, command: Command) {
    if tx.push(command).is_err() {
        eprintln!("warning: audio command queue full");
    }
}

/// Make segments until told to stop.
///
/// Each segment continues from the previous one. A tweak (a change of `version`) cancels the segment being made and
/// discards a finished one that has not started; the next one then continues from the segment playing now.
fn generate_loop(
    backend: Backend,
    rate: u32,
    shared: Arc<Mutex<Shared>>,
    version: Arc<AtomicU64>,
    station: Option<Station>,
    quit: Arc<AtomicBool>,
) {
    let result = (|| -> Result<()> {
        let env = ort::environment::init().with_name("driftscape").with_telemetry(false).build()?;
        let mut voice = MusicGen::load(&env, &models_dir()?.join("musicgen-small"), backend)?;
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
            let started = std::time::Instant::now();
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
            let label = format!(
                "segment {} · {} · {} · {:.0} s in {:.1} s ({:.1}× real time)",
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
        eprintln!("generator stopped: {e:#}");
        quit.store(true, Ordering::Release);
    }
}

/// Where a segment should jump back to if it has to repeat: a point whose sound matches its end, at least a few
/// seconds back so repeats are not short, and on a bar line when the genre has a beat.
fn rejoin_for(audio: &Audio, genre: &Genre, rate: u32, overlap: usize, crossfade: usize) -> usize {
    let samples = &audio.channels[0];
    let from = samples.len() - crossfade;
    let min_loop = 4 * rate as usize;
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

fn listener_or_default<'a>(listener: &'a str, genre: &'a Genre) -> &'a str {
    if listener.trim().is_empty() { &genre.default_prompt } else { listener }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_owned() } else { format!("{}…", s.chars().take(n - 1).collect::<String>()) }
}

/// Read tweaks from the terminal, one per line.
fn spawn_input(shared: Arc<Mutex<Shared>>, version: Arc<AtomicU64>, quit: Arc<AtomicBool>, radio: bool) {
    thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if std::env::var_os("DRIFTSCAPE_TRACE").is_some() {
                eprintln!("[trace] input {line:?}");
            }
            if line.is_empty() {
                continue;
            }
            // Quitting must never wait on anything.
            if matches!(line, ":q" | ":quit") {
                quit.store(true, Ordering::Release);
                break;
            }
            // Messages are printed after the lock is released, so console output never holds up the other threads.
            let mut messages: Vec<String> = Vec::new();
            macro_rules! say {
                ($($arg:tt)*) => { messages.push(format!($($arg)*)) };
            }
            let mut guard = shared.lock().unwrap();
            let before =
                (guard.params.genre.id.clone(), guard.params.prompt.clone(), guard.params.focus, guard.params.variety);
            let p = &mut guard.params;
            let mut words = line.split_whitespace();
            match words.next().unwrap_or("") {
                ":help" => say!(
                    "  <text>            new prompt (next segment)\n  :genre <id>       switch genre: {}\n  :focus + | -      follow the prompt more or less closely\n  :variety + | -    let the music wander more or less\n  :status           show the current settings\n  :quit             stop and save the recording",
                    builtin_genres().iter().map(|g| g.id.as_str()).collect::<Vec<_>>().join(", ")
                ),
                ":genre" if !radio => {
                    match words.next().and_then(|id| builtin_genres().into_iter().find(|g| g.id == id)) {
                        Some(g) => {
                            say!("  genre → {}", g.name);
                            p.genre = g;
                        }
                        None => say!("  unknown genre; :help lists them"),
                    }
                }
                ":focus" => {
                    p.focus += if words.next() == Some("-") { -0.5 } else { 0.5 };
                    say!("  focus {:+.1}", p.focus);
                }
                ":variety" => {
                    p.variety += if words.next() == Some("-") { -0.1 } else { 0.1 };
                    say!("  variety {:+.1}", p.variety);
                }
                ":status" => say!(
                    "  {} · {} · focus {:+.1} · variety {:+.1}",
                    p.genre.name,
                    listener_or_default(&p.prompt, &p.genre),
                    p.focus,
                    p.variety
                ),
                _ if line.starts_with(':') => say!("  unknown command; :help lists them"),
                _ if radio => say!("  radio plays its station's prompts; use `driftscape live` to steer"),
                _ => {
                    p.prompt = line.to_owned();
                    say!("  prompt → {line}");
                }
            }
            let p = &guard.params;
            if (p.genre.id.clone(), p.prompt.clone(), p.focus, p.variety) != before {
                // Start over from what is playing: drop the waiting segment, cancel the one being made.
                guard.ready = None;
                version.fetch_add(1, Ordering::AcqRel);
                say!("  (generating with the change; you'll hear it in about one segment's time)");
            }
            drop(guard);
            for m in messages {
                eprintln!("{m}");
            }
        }
    });
}
