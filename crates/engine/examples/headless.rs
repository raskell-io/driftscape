//! Run a live session without an audio device, the way a plugin host drives it: call `process` in blocks at real-time
//! pace, change the prompt part-way, and write what was produced to a WAV file. Nothing plays.
//!
//! cargo run --release -p driftscape-engine --example headless -- <out.wav> [seconds] [tweak-at-seconds] [bed]
//! (the left channel is analysed and written)

use driftscape_core::{
    Audio,
    preset::builtin_genres,
    wav::{SampleFormat, encode},
};
use driftscape_engine::{Config, Event, Params, Session, VoiceBackend};
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).expect("output path");
    let seconds: f64 = args.get(2).map_or(40.0, |s| s.parse().unwrap());
    let tweak_at: f64 = args.get(3).map_or(20.0, |s| s.parse().unwrap());
    let rate = 48_000;
    let genre = builtin_genres().into_iter().find(|g| g.id == "ambient").unwrap();
    let params = Params::new(genre, "dusk over a quiet harbour");
    let bed = std::env::args().nth(4).and_then(|k| driftscape_core::ambience::Kind::from_id(&k));
    let (session, mut audio) = Session::start(Config {
        bed: bed.map(|kind| driftscape_engine::BedChoice { kind, db: 0.0, intensity: 0.6 }),
        params,
        station: None,
        backend: VoiceBackend::Gpu,
        rate,
    })?;

    let block = 512;
    let mut samples = Vec::new();
    let mut buf = vec![0.0f32; block];
    let mut right = vec![0.0f32; block];
    let started = Instant::now();
    let mut tweaked = false;
    let mut first_sound = None;
    while (samples.len() as f64) < seconds * f64::from(rate) {
        // Real-time pace: never run ahead of the wall clock.
        let due = Duration::from_secs_f64(samples.len() as f64 / f64::from(rate));
        if let Some(wait) = due.checked_sub(started.elapsed()) {
            std::thread::sleep(wait);
        }
        audio.process(&mut buf, &mut right);
        if first_sound.is_none() && buf.iter().any(|s| s.abs() > 1e-4) {
            first_sound = Some(samples.len());
        }
        samples.extend_from_slice(&buf);
        let t = samples.len() as f64 / f64::from(rate);
        if !tweaked && first_sound.is_some_and(|f| t - f as f64 / f64::from(rate) >= tweak_at) {
            tweaked = true;
            session.update(|p| p.prompt = "deep cave, slow drips, distant drone".into());
            println!("{t:6.1} s  tweak: new prompt");
        }
        while let Some(e) = session.try_event() {
            match e {
                Event::Trace(_) => {}
                e => println!("{t:6.1} s  {e:?}"),
            }
        }
        if !session.is_running() {
            break;
        }
    }
    let from = first_sound.unwrap_or(0);
    let played = &samples[from..];
    // The biggest jump between neighbouring samples, and the longest near-silent stretch once music started.
    let max_step = played.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
    let (mut quiet, mut longest) = (0usize, 0usize);
    for s in played {
        quiet = if s.abs() < 1e-3 { quiet + 1 } else { 0 };
        longest = longest.max(quiet);
    }
    println!(
        "first sound after {:.1} s; {:.1} s of music; largest sample step {max_step:.3}; longest near-silence {:.2} s",
        from as f64 / f64::from(rate),
        played.len() as f64 / f64::from(rate),
        longest as f64 / f64::from(rate)
    );
    std::fs::write(out, encode(&Audio::mono(rate, samples), SampleFormat::Pcm16))?;
    Ok(())
}
