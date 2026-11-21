//! Live and radio mode in the terminal: the session engine ([`driftscape_engine`]) played through the default audio
//! output, steered by lines typed while it plays, and recorded to a WAV file.
//!
//! Threads besides the engine's own: the **audio callback** (cpal) calls the engine's real-time side; the
//! **recorder** streams everything played to disk; the **input** thread forwards typed lines to the main loop, which
//! applies them and prints what the engine reports.

use crate::recorder::WavRecorder;
use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use driftscape_core::preset::{Genre, Station, builtin_genres};
use driftscape_engine::{Config, Event, Params, Session, listener_or_default};
use driftscape_infer::musicgen::Backend;
use std::{
    io::BufRead,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
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

    let (params, station) = match options.source {
        Source::Live { genre, prompt } => (Params::new(genre, prompt), None),
        Source::Radio { station, genre } => (Params::new(genre, ""), Some(station)),
    };
    let radio = station.is_some();
    match &station {
        Some(s) => eprintln!("📻 {} · {}", s.name, s.description),
        None => eprintln!("live · {} · {}", params.genre.name, listener_or_default(&params.prompt, &params.genre)),
    }
    let (mut session, mut audio) = Session::start(Config { params, station, backend: options.backend, rate })?;
    {
        // The first Ctrl-C stops cleanly (and saves the recording); a second one exits at once, whatever is stuck.
        let stopper = session.stopper();
        ctrlc::set_handler(move || {
            if stopper.swap() {
                eprintln!("\nstopped");
                std::process::exit(130);
            }
        })
        .context("installing the Ctrl-C handler")?;
    }

    // Audio → recorder: the samples played.
    let (mut record_tx, mut record_rx) = rtrb::RingBuffer::<f32>::new(rate as usize * 4);
    let recording = options.record.is_some();
    let mut scratch = vec![0.0f32; 8192];
    let stream = device
        .build_output_stream(
            config,
            move |out: &mut [f32], _| {
                for frames in out.chunks_mut(scratch.len() * channels) {
                    let mono = &mut scratch[..frames.len() / channels];
                    audio.process(mono);
                    for (frame, &s) in frames.chunks_mut(channels).zip(mono.iter()) {
                        frame.fill(s);
                    }
                    if recording {
                        for &s in mono.iter() {
                            let _ = record_tx.push(s);
                        }
                    }
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .context("opening the audio stream")?;
    stream.play().context("starting playback")?;

    let done = Arc::new(AtomicBool::new(false));
    let recorder = match &options.record {
        Some(path) => {
            let mut wav = WavRecorder::create(path, rate)?;
            let done = done.clone();
            Some(thread::Builder::new().name("recorder".into()).spawn(move || -> Result<WavRecorder> {
                let mut chunk = Vec::with_capacity(rate as usize);
                let mut started = false; // skip the silence while the first segment is generated
                loop {
                    let finished = done.load(Ordering::Acquire);
                    while let Ok(s) = record_rx.pop() {
                        started |= s.abs() > 1e-4;
                        if started {
                            chunk.push(s);
                        }
                    }
                    wav.write(&chunk)?;
                    chunk.clear();
                    if finished {
                        return Ok(wav);
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            })?)
        }
        None => None,
    };

    let lines = spawn_input();
    eprintln!("type a new prompt and press Enter to steer; :help lists commands; Ctrl-C stops");
    if let Some(path) = &options.record {
        eprintln!("recording to {}", path.display());
    }
    eprintln!("loading the model…");

    let deadline = options.stop_after.map(|m| Instant::now() + Duration::from_secs_f64(m * 60.0));
    let trace = std::env::var_os("DRIFTSCAPE_TRACE").is_some();
    while session.is_running() {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        while let Some(event) = session.try_event() {
            match event {
                Event::Loaded => eprintln!("generating the first segment…"),
                Event::Playing { label } => eprintln!("▶ {label}"),
                Event::Repeating => eprintln!("↺ repeating while the next segment is generated"),
                Event::Paused => eprintln!(
                    "  the audio output stopped (did the output device change?); Ctrl-C to stop, or reconnect the device"
                ),
                Event::Resumed => eprintln!("  audio output resumed"),
                Event::Failed(e) => eprintln!("generator stopped: {e}"),
                Event::Trace(t) if trace => eprintln!("[trace] {t}"),
                Event::Trace(_) => {}
            }
        }
        while let Ok(line) = lines.try_recv() {
            if trace {
                eprintln!("[trace] input {line:?}");
            }
            if matches!(line.as_str(), ":q" | ":quit") {
                session.stopper().stop();
                break;
            }
            for message in apply(&line, &session, radio) {
                eprintln!("{message}");
            }
        }
        thread::sleep(Duration::from_millis(30));
    }

    eprintln!("\nstopping…");
    drop(stream);
    session.stop();
    done.store(true, Ordering::Release);
    if let Some(recorder) = recorder {
        let wav = recorder.join().map_err(|_| anyhow!("recorder thread panicked"))??;
        let seconds = wav.seconds(rate);
        let path = wav.finish()?;
        eprintln!("saved {:.0} s of this session to {}", seconds, path.display());
    }
    Ok(())
}

/// Lines typed in the terminal, trimmed, empty ones skipped.
fn spawn_input() -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if !line.is_empty() && tx.send(line.to_owned()).is_err() {
                break;
            }
        }
    });
    rx
}

/// Apply one typed line; returns what to print.
fn apply(line: &str, session: &Session, radio: bool) -> Vec<String> {
    let mut messages = Vec::new();
    let mut words = line.split_whitespace();
    let command = words.next().unwrap_or("");
    let changed = match command {
        ":help" => {
            messages.push(format!(
                "  <text>            new prompt (next segment)\n  :genre <id>       switch genre: {}\n  :focus + | -      follow the prompt more or less closely\n  :variety + | -    let the music wander more or less\n  :status           show the current settings\n  :quit             stop and save the recording",
                builtin_genres().iter().map(|g| g.id.as_str()).collect::<Vec<_>>().join(", ")
            ));
            false
        }
        ":genre" if !radio => match words.next().and_then(|id| builtin_genres().into_iter().find(|g| g.id == id)) {
            Some(g) => {
                messages.push(format!("  genre → {}", g.name));
                session.update(|p| p.genre = g)
            }
            None => {
                messages.push("  unknown genre; :help lists them".into());
                false
            }
        },
        ":focus" | ":variety" => {
            let up = words.next() != Some("-");
            let mut now = 0.0;
            let changed = session.update(|p| {
                let (value, step) = if command == ":focus" { (&mut p.focus, 0.5) } else { (&mut p.variety, 0.1) };
                *value += if up { step } else { -step };
                now = *value;
            });
            messages.push(format!("  {} {now:+.1}", &command[1..]));
            changed
        }
        ":status" => {
            let p = session.params();
            messages.push(format!(
                "  {} · {} · focus {:+.1} · variety {:+.1}",
                p.genre.name,
                listener_or_default(&p.prompt, &p.genre),
                p.focus,
                p.variety
            ));
            false
        }
        _ if line.starts_with(':') => {
            messages.push("  unknown command; :help lists them".into());
            false
        }
        _ if radio => {
            messages.push("  radio plays its station's prompts; use `driftscape live` to steer".into());
            false
        }
        _ => {
            messages.push(format!("  prompt → {line}"));
            session.update(|p| p.prompt = line.to_owned())
        }
    };
    if changed {
        messages.push("  (generating with the change; you'll hear it in about one segment's time)".into());
    }
    messages
}
