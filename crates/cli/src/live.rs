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
    Request, Tokens, Voice,
    musicgen::{Backend, MusicGen},
};
use std::{
    io::BufRead,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
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

/// A finished segment, at the output rate.
struct Segment {
    buffer: Buffer,
    /// Frames at the start that replay the previous segment's end.
    overlap: usize,
    /// Crossfade into this segment, and out of it when it repeats.
    crossfade: usize,
    /// Where to jump back to if this segment has to repeat: its last `crossfade` frames sound like the ones from here.
    rejoin: usize,
    label: String,
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
    let config: cpal::StreamConfig = config.into();
    let rate = config.sample_rate;
    let channels = usize::from(config.channels);

    let (genre, prompt, station) = match options.source {
        Source::Live { genre, prompt } => (genre, prompt, None),
        Source::Radio { station, genre } => (genre, String::new(), Some(station)),
    };
    let params = Arc::new(Mutex::new(Params { genre, prompt, focus: 0.0, variety: 0.0 }));
    let quit = Arc::new(AtomicBool::new(false));
    {
        let quit = quit.clone();
        ctrlc::set_handler(move || quit.store(true, Ordering::SeqCst)).context("installing the Ctrl-C handler")?;
    }

    // Generator → control: one finished segment waits while the next is being made.
    let (segment_tx, segment_rx) = mpsc::sync_channel::<Segment>(1);
    let generator = {
        let (params, quit, station) = (params.clone(), quit.clone(), station.clone());
        thread::Builder::new()
            .name("generator".into())
            .spawn(move || generate_loop(options.backend, rate, params, station, segment_tx, quit))?
    };

    // Control → audio: commands; audio → control: finished buffers to drop; audio → recorder: played samples.
    let (mut command_tx, mut command_rx) = rtrb::RingBuffer::<Command>::new(64);
    let (mut finished_tx, mut finished_rx) = rtrb::RingBuffer::<Buffer>::new(64);
    let (mut record_tx, mut record_rx) = rtrb::RingBuffer::<f32>::new(rate as usize * 4);
    let clock = Arc::new(AtomicU64::new(0));
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

    spawn_input(params.clone(), quit.clone(), station.is_some());
    match &station {
        Some(s) => eprintln!("📻 {} · {}", s.name, s.description),
        None => {
            let p = params.lock().unwrap();
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
        match &playing {
            None => {
                // Wait for the first segment, then start it with a short fade-in.
                if let Ok(segment) = segment_rx.recv_timeout(Duration::from_millis(50)) {
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
                continue;
            }
            Some(current) => {
                let fade = current.segment.crossfade;
                let at = current.end() - fade as u64;
                if now + lookahead >= at {
                    let id = next_id;
                    next_id += 1;
                    let next = match segment_rx.try_recv() {
                        Ok(segment) => {
                            // The new segment's frame `overlap - fade` sounds like the current one's `len - fade`.
                            let offset = segment.overlap.saturating_sub(fade);
                            eprintln!("▶ {}", segment.label);
                            Playing { id, segment: Arc::new(segment), start: at, offset }
                        }
                        Err(_) => {
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
    // Let the generator notice and finish; it may be mid-segment.
    drop(segment_rx);
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

fn push(tx: &mut rtrb::Producer<Command>, command: Command) {
    if tx.push(command).is_err() {
        eprintln!("warning: audio command queue full");
    }
}

/// Make segments until told to stop.
fn generate_loop(
    backend: Backend,
    rate: u32,
    params: Arc<Mutex<Params>>,
    station: Option<Station>,
    segments: mpsc::SyncSender<Segment>,
    quit: Arc<AtomicBool>,
) {
    let result = (|| -> Result<()> {
        let env = ort::environment::init().with_name("driftscape").with_telemetry(false).build()?;
        let mut voice = MusicGen::load(&env, &models_dir()?.join("musicgen-small"), backend)?;
        let mut previous: Option<Tokens> = None;
        let mut gain = None::<f32>;
        let mut seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos() as u64;
        for number in 0.. {
            if quit.load(Ordering::Acquire) {
                break;
            }
            let p = params.lock().unwrap().clone();
            let listener = match &station {
                Some(s) => s.prompt_for(number),
                None => &p.prompt,
            };
            let g = &p.genre;
            let continue_from = previous.as_ref().map(|t| t.tail(g.overlap_seconds));
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let request = Request {
                guidance: (g.guidance + p.focus).clamp(1.0, 8.0),
                temperature: (g.temperature + p.variety).clamp(0.5, 1.6),
                continue_from,
                ..Request::new(g.prompt(listener), g.segment_seconds, seed)
            };
            let started = std::time::Instant::now();
            let clip = voice.generate(&request, &mut |_| {})?;
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
                "segment {} · {} · {:.0} s in {:.1} s ({:.1}× real time)",
                number + 1,
                truncate(listener_or_default(listener, g), 48),
                g.segment_seconds,
                took,
                g.segment_seconds / took
            );
            previous = Some(clip.tokens);
            if segments.send(Segment { buffer, overlap, crossfade, rejoin, label }).is_err() {
                break; // the session ended
            }
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
fn spawn_input(params: Arc<Mutex<Params>>, quit: Arc<AtomicBool>, radio: bool) {
    thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut p = params.lock().unwrap();
            let mut words = line.split_whitespace();
            match words.next().unwrap_or("") {
                ":q" | ":quit" => {
                    quit.store(true, Ordering::Release);
                    break;
                }
                ":help" => eprintln!(
                    "  <text>            new prompt (next segment)\n  :genre <id>       switch genre: {}\n  :focus + | -      follow the prompt more or less closely\n  :variety + | -    let the music wander more or less\n  :status           show the current settings\n  :quit             stop and save the recording",
                    builtin_genres().iter().map(|g| g.id.as_str()).collect::<Vec<_>>().join(", ")
                ),
                ":genre" if !radio => {
                    match words.next().and_then(|id| builtin_genres().into_iter().find(|g| g.id == id)) {
                        Some(g) => {
                            eprintln!("  genre → {} (from the next segment)", g.name);
                            p.genre = g;
                        }
                        None => eprintln!("  unknown genre; :help lists them"),
                    }
                }
                ":focus" => {
                    p.focus += if words.next() == Some("-") { -0.5 } else { 0.5 };
                    eprintln!("  focus {:+.1}", p.focus);
                }
                ":variety" => {
                    p.variety += if words.next() == Some("-") { -0.1 } else { 0.1 };
                    eprintln!("  variety {:+.1}", p.variety);
                }
                ":status" => eprintln!(
                    "  {} · {} · focus {:+.1} · variety {:+.1}",
                    p.genre.name,
                    listener_or_default(&p.prompt, &p.genre),
                    p.focus,
                    p.variety
                ),
                _ if line.starts_with(':') => eprintln!("  unknown command; :help lists them"),
                _ if radio => eprintln!("  radio plays its station's prompts; use `driftscape live` to steer"),
                _ => {
                    p.prompt = line.to_owned();
                    eprintln!("  prompt → {line} (from the next segment)");
                }
            }
        }
    });
}
