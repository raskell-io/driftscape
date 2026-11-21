//! `driftscape`: the command-line way into the engine.

mod generate;
mod play;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use driftscape_core::{
    Audio,
    looping::{LoopSpec, make_loop},
    resample::resample,
    tempo,
    wav::{self, SampleFormat},
};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "driftscape", version, about = "Ambient soundscapes, generated on your own machine.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a clip from a text prompt, on this machine.
    Gen {
        /// What it should sound like.
        prompt: String,
        /// Where to write the clip.
        #[arg(short, long)]
        output: PathBuf,
        /// Length in seconds (up to 30).
        #[arg(long, default_value_t = 10.0)]
        seconds: f32,
        /// Random seed. The same seed and prompt give the same clip.
        #[arg(long)]
        seed: Option<u64>,
        /// How strongly to follow the prompt.
        #[arg(long, default_value_t = 3.0)]
        guidance: f32,
        /// Run the model on the GPU or the CPU.
        #[arg(long, value_enum, default_value_t = Device::Gpu)]
        device: Device,
    },
    /// Turn an audio clip into a seamless loop.
    ///
    /// With --bars the loop is a whole number of bars at the clip's tempo (detected, or given with --bpm).
    /// With --seconds it is that long. Without either, it is as long as the clip allows.
    Loop {
        /// Input WAV file.
        input: PathBuf,
        /// Where to write the loop.
        #[arg(short, long)]
        output: PathBuf,
        /// Loop length in bars of 4/4.
        #[arg(long, conflicts_with = "seconds")]
        bars: Option<u32>,
        /// Tempo in BPM, when detection gets it wrong or the clip has no clear pulse.
        #[arg(long, requires = "bars")]
        bpm: Option<f64>,
        /// Loop length in seconds.
        #[arg(long)]
        seconds: Option<f64>,
        /// Crossfade at the seam, in milliseconds.
        #[arg(long, default_value_t = 250.0)]
        crossfade_ms: f64,
        /// Output sample rate, e.g. 48000 for a DAW session. Defaults to the input's.
        #[arg(long)]
        rate: Option<u32>,
        /// Output sample format.
        #[arg(long, value_enum, default_value_t = Format::Pcm24)]
        format: Format,
    },
    /// Play a WAV file on repeat until you press Ctrl-C.
    Play { input: PathBuf },
    /// Print a clip's length and detected tempo.
    Info { input: PathBuf },
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Device {
    Gpu,
    Cpu,
}

impl From<Device> for driftscape_infer::musicgen::Backend {
    fn from(d: Device) -> Self {
        match d {
            Device::Gpu => Self::Gpu,
            Device::Cpu => Self::Cpu,
        }
    }
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Format {
    Pcm16,
    Pcm24,
    Float32,
}

impl From<Format> for SampleFormat {
    fn from(f: Format) -> Self {
        match f {
            Format::Pcm16 => SampleFormat::Pcm16,
            Format::Pcm24 => SampleFormat::Pcm24,
            Format::Float32 => SampleFormat::Float32,
        }
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Gen { prompt, output, seconds, seed, guidance, device } => {
            let seed = seed.unwrap_or_else(|| {
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
            });
            generate::run(&prompt, seconds, seed, guidance, device.into(), &output)?;
        }
        Command::Loop { input, output, bars, bpm, seconds, crossfade_ms, rate, format } => {
            let clip = read(&input)?;
            let crossfade = crossfade_ms / 1000.0;
            let spec = match (bars, seconds) {
                (Some(bars), _) => {
                    let bpm = match bpm {
                        Some(bpm) => bpm,
                        None => detect(&clip)?.bpm,
                    };
                    let spec = LoopSpec::bars(bars, 4, bpm, clip.sample_rate, crossfade);
                    eprintln!("{bars} bars at {bpm} BPM = {:.2} s", spec.length as f64 / f64::from(clip.sample_rate));
                    spec
                }
                (None, Some(s)) => LoopSpec::seconds(s, clip.sample_rate, crossfade),
                (None, None) => {
                    let spec = LoopSpec::seconds(0.0, clip.sample_rate, crossfade);
                    LoopSpec { length: clip.len().saturating_sub(spec.crossfade), ..spec }
                }
            };
            let looped = make_loop(&clip, spec)?;
            eprintln!(
                "loop starts at {:.3} s, seam similarity {:.2}",
                looped.start as f64 / f64::from(clip.sample_rate),
                looped.seam_similarity
            );
            let audio = match rate {
                Some(r) => resample(&looped.audio, r),
                None => looped.audio,
            };
            write(&output, &audio, format.into())?;
            eprintln!("wrote {} ({:.2} s, {} Hz)", output.display(), audio.seconds(), audio.sample_rate);
        }
        Command::Play { input } => {
            let audio = read(&input)?;
            eprintln!("playing {} on repeat ({:.2} s); Ctrl-C to stop", input.display(), audio.seconds());
            play::looped(&audio)?;
        }
        Command::Info { input } => {
            let clip = read(&input)?;
            println!("{}", input.display());
            println!(
                "  {:.2} s, {} Hz, {} channel(s), peak {:.2}",
                clip.seconds(),
                clip.sample_rate,
                clip.channel_count(),
                clip.peak()
            );
            match tempo::estimate(&clip.mixdown(), clip.sample_rate, tempo::Range::default()) {
                Some(t) => println!("  tempo {} BPM (confidence {:.2})", t.bpm, t.confidence),
                None => println!("  no steady pulse"),
            }
        }
    }
    Ok(())
}

fn read(path: &Path) -> Result<Audio> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    wav::decode(&bytes).with_context(|| format!("decoding {}", path.display()))
}

pub(crate) fn write(path: &Path, audio: &Audio, format: SampleFormat) -> Result<()> {
    std::fs::write(path, wav::encode(audio, format)).with_context(|| format!("writing {}", path.display()))
}

fn detect(clip: &Audio) -> Result<tempo::Tempo> {
    match tempo::estimate(&clip.mixdown(), clip.sample_rate, tempo::Range::default()) {
        Some(t) => {
            eprintln!("detected {} BPM (confidence {:.2})", t.bpm, t.confidence);
            Ok(t)
        }
        None => bail!("this clip has no steady pulse; pass --bpm, or use --seconds instead of --bars"),
    }
}
