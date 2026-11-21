//! `driftscape gen`: load a voice from the model cache and generate a clip.

use anyhow::{Context, Result, bail};
use driftscape_core::wav::SampleFormat;
use driftscape_infer::{
    Request, Voice,
    musicgen::{Backend, MusicGen},
};
use std::{
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

/// Where models live: `~/Library/Application Support/Driftscape/models` on macOS, `~/.local/share/driftscape/models`
/// on Linux. The CLI and the plugin share it.
pub fn models_dir() -> Result<PathBuf> {
    let base = dirs::data_dir().context("no data directory for this user")?;
    let app = if cfg!(target_os = "macos") { "Driftscape" } else { "driftscape" };
    Ok(base.join(app).join("models"))
}

pub fn run(prompt: &str, seconds: f32, seed: u64, guidance: f32, backend: Backend, output: &Path) -> Result<()> {
    let dir = models_dir()?.join("musicgen-small");
    if !dir.join("tokenizer.json").exists() {
        bail!("MusicGen is not in {}; download Xenova/musicgen-small (q4 ONNX files) there", dir.display());
    }

    let started = Instant::now();
    let env = ort::environment::init().with_name("driftscape").with_telemetry(false).build()?;
    let mut voice = MusicGen::load(&env, &dir, backend)?;
    let loaded = started.elapsed();
    eprintln!("loaded MusicGen ({backend:?}) in {:.1} s", loaded.as_secs_f32());

    let request = Request { guidance, ..Request::new(prompt, seconds, seed) };
    let started = Instant::now();
    let audio = voice.generate(&request, &mut |p| {
        if p.step % 25 == 0 || p.step == p.total {
            let elapsed = started.elapsed().as_secs_f32();
            eprint!("\rgenerating {:>3}%  {:>5.1} s", p.step * 100 / p.total, elapsed);
            let _ = std::io::stderr().flush();
        }
    })?;
    let took = started.elapsed().as_secs_f32();
    eprintln!();
    eprintln!(
        "{:.1} s of audio in {:.1} s: {:.2}× real time (seed {seed})",
        audio.seconds(),
        took,
        audio.seconds() as f32 / took
    );
    crate::write(output, &audio, SampleFormat::Pcm24)?;
    eprintln!("wrote {}", output.display());
    Ok(())
}
