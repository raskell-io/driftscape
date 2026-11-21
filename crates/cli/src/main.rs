//! `driftscape`: the command-line way into the engine.

mod generate;
mod live;
mod play;
mod recorder;
mod serve;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use driftscape_core::{
    Audio,
    arrange::{self, Arrangement},
    looping::{LoopSpec, fit_loop, make_loop},
    preset,
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
    ///
    /// With --bpm and --bars the result is a seamless loop of exactly that many bars at exactly that tempo, ready to
    /// drop into a DAW session: the model is asked for the tempo, then the clip is measured, stretched and cut.
    Gen {
        /// What it should sound like.
        prompt: String,
        /// Where to write the clip.
        #[arg(short, long)]
        output: PathBuf,
        /// Length in seconds (up to 30). Ignored with --bars.
        #[arg(long, default_value_t = 10.0)]
        seconds: f32,
        /// Tempo to ask for, and to fit the loop to with --bars.
        #[arg(long)]
        bpm: Option<f64>,
        /// Make a seamless loop of this many bars of 4/4 at --bpm.
        #[arg(long, requires = "bpm")]
        bars: Option<u32>,
        /// Output sample rate, e.g. 48000 for a DAW session. Defaults to the model's 32000.
        #[arg(long)]
        rate: Option<u32>,
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
    /// Play endless music, generated as it plays. Type while it plays to steer it.
    Live {
        /// What it should sound like. Defaults to the genre's own prompt.
        prompt: Vec<String>,
        /// Genre preset: ambient, synthetic-ambience, realistic-ambience, ambient-techno, thematic.
        #[arg(short, long, default_value = "ambient")]
        genre: String,
        #[command(flatten)]
        session: SessionArgs,
    },
    /// Tune in to a station. Without a name, list the stations.
    Radio {
        station: Option<String>,
        #[command(flatten)]
        session: SessionArgs,
    },
    /// Open the web app on this machine, with the native engine behind it (faster than in the browser).
    Serve {
        #[arg(long, default_value_t = 4321)]
        port: u16,
        /// Run the model on the GPU or the CPU.
        #[arg(long, value_enum, default_value_t = Device::Gpu)]
        device: Device,
        /// Don't open a browser.
        #[arg(long)]
        no_open: bool,
    },
    /// List the genres and stations.
    Presets {
        /// Print them as JSON (what the web app loads).
        #[arg(long)]
        json: bool,
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
        /// The clip's tempo in BPM, when detection gets it wrong or the clip has no clear pulse.
        #[arg(long, requires = "bars")]
        bpm: Option<f64>,
        /// Stretch the clip to this tempo first, so the loop is whole bars at exactly this BPM.
        #[arg(long, requires = "bars")]
        to_bpm: Option<f64>,
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
    /// Mix a producer-mode project down to a WAV file.
    ///
    /// The project is JSON: `{"sample_rate": 48000, "sources": ["a.wav", ...], "tracks": [{"gain_db": 0, "pan": 0,
    /// "clips": [{"source": 0, "at": 0, "offset": 0, "length": 8, "fade_in": 2, "fade_out": 2}]}]}`. Source paths are
    /// relative to the project file.
    Render {
        project: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Output sample format.
        #[arg(long, value_enum, default_value_t = Format::Pcm24)]
        format: Format,
    },
    /// Play a WAV file on repeat until you press Ctrl-C.
    Play { input: PathBuf },
    /// Print a clip's length and detected tempo.
    Info { input: PathBuf },
}

#[derive(clap::Args)]
struct SessionArgs {
    /// Run the model on the GPU or the CPU.
    #[arg(long, value_enum, default_value_t = Device::Gpu)]
    device: Device,
    /// Where to save the recording of the session. Defaults to ~/Music/Driftscape.
    #[arg(long, conflicts_with = "no_record")]
    record: Option<PathBuf>,
    /// Don't record the session.
    #[arg(long)]
    no_record: bool,
    /// Stop by itself after this many minutes, like a sleep timer.
    #[arg(long)]
    stop_after: Option<f64>,
    /// An ambience bed under the music: rain, wind, waves or fire; auto picks one from the prompt; off. Defaults to
    /// the genre's (realistic-ambience: auto).
    #[arg(long)]
    bed: Option<String>,
    /// The bed's level in dB; 0 sits a few dB under the music.
    #[arg(long, allow_hyphen_values = true)]
    bed_level: Option<f32>,
}

impl SessionArgs {
    fn bed(&self) -> Result<live::BedMode> {
        Ok(match self.bed.as_deref() {
            None => live::BedMode::Genre,
            Some(word) => live::BedMode::parse(word)
                .with_context(|| format!("no bed `{word}`; choose rain, wind, waves, fire, auto or off"))?,
        })
    }

    fn record_path(&self, name: &str) -> Option<PathBuf> {
        if self.no_record {
            return None;
        }
        self.record.clone().or_else(|| {
            let dir = dirs::audio_dir().or_else(dirs::home_dir)?.join("Driftscape");
            Some(dir.join(format!("driftscape-{name}-{}.wav", recorder::timestamp())))
        })
    }
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
        Command::Live { prompt, genre, session } => {
            let genre = find_genre(&genre)?;
            let record = session.record_path(&genre.id);
            live::run(live::Options {
                source: live::Source::Live { genre, prompt: prompt.join(" ") },
                backend: session.device.into(),
                record,
                stop_after: session.stop_after,
                bed: session.bed()?,
                bed_db: session.bed_level,
            })?;
        }
        Command::Serve { port, device, no_open } => serve::run(port, device.into(), !no_open)?,
        Command::Presets { json } => {
            let (genres, stations) = (preset::builtin_genres(), preset::builtin_stations());
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({ "genres": genres, "stations": stations }))?
                );
            } else {
                println!("genres:");
                for g in &genres {
                    println!("  {:<20} {}", g.id, g.description);
                }
                println!("stations:");
                for s in &stations {
                    println!("  {:<20} {} ({})", s.id, s.description, s.genre);
                }
            }
        }
        Command::Radio { station, session } => {
            let stations = preset::builtin_stations();
            let Some(id) = station else {
                println!("stations:");
                for s in &stations {
                    println!("  {:<14} {} ({})", s.id, s.description, s.genre);
                }
                return Ok(());
            };
            let station = stations
                .into_iter()
                .find(|s| s.id == id)
                .with_context(|| format!("no station `{id}`; run `driftscape radio` to list them"))?;
            let genre = find_genre(&station.genre)?;
            let record = session.record_path(&station.id);
            live::run(live::Options {
                source: live::Source::Radio { station, genre },
                backend: session.device.into(),
                record,
                stop_after: session.stop_after,
                bed: session.bed()?,
                bed_db: session.bed_level,
            })?;
        }
        Command::Gen { prompt, output, seconds, bpm, bars, rate, seed, guidance, device } => {
            let seed = seed.unwrap_or_else(|| {
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
            });
            let prompt = match bpm {
                Some(b) => format!("{prompt}, {b:.0} BPM"),
                None => prompt,
            };
            let audio = match (bars, bpm) {
                (Some(bars), Some(bpm)) => {
                    // Enough for the bars after a stretch of up to 25 %, plus room to choose a start and crossfade.
                    let needed = f64::from(bars) * 4.0 * 60.0 / bpm;
                    let seconds = (needed * 1.25 + 1.5).min(30.0) as f32;
                    if needed + 0.5 > 30.0 {
                        bail!("{bars} bars at {bpm} BPM is {needed:.1} s; MusicGen makes at most 30 s");
                    }
                    let clip = generate::run(&prompt, seconds, seed, guidance, device.into())?;
                    let fit = fit_loop(&clip, bpm, bars, 4, 0.05, None).context(
                        "fitting the clip to the tempo (try another --seed, or a prompt with a clearer beat)",
                    )?;
                    eprintln!(
                        "played at {:.2} BPM, stretched by {:.2} % to {bpm} BPM; {bars} bars = {:.2} s, seam similarity {:.2}",
                        fit.played_bpm,
                        (fit.ratio - 1.0) * 100.0,
                        fit.looped.audio.seconds(),
                        fit.looped.seam_similarity
                    );
                    fit.looped.audio
                }
                _ => generate::run(&prompt, seconds, seed, guidance, device.into())?,
            };
            let mut audio = match rate {
                Some(r) => resample(&audio, r),
                None => audio,
            };
            // The model's output can exceed full scale; bring the peak to -1 dBFS so nothing clips.
            audio.normalize(0.89);
            write(&output, &audio, SampleFormat::Pcm24)?;
            eprintln!("wrote {} ({:.2} s, {} Hz)", output.display(), audio.seconds(), audio.sample_rate);
        }
        Command::Loop { input, output, bars, bpm, to_bpm, seconds, crossfade_ms, rate, format } => {
            let clip = read(&input)?;
            let crossfade = crossfade_ms / 1000.0;
            if let (Some(bars), Some(target)) = (bars, to_bpm) {
                let fit = fit_loop(&clip, target, bars, 4, crossfade, bpm)?;
                eprintln!(
                    "played at {:.2} BPM, stretched by {:.2} % to {target} BPM; seam similarity {:.2}",
                    fit.played_bpm,
                    (fit.ratio - 1.0) * 100.0,
                    fit.looped.seam_similarity
                );
                let audio = match rate {
                    Some(r) => resample(&fit.looped.audio, r),
                    None => fit.looped.audio,
                };
                write(&output, &audio, format.into())?;
                eprintln!("wrote {} ({:.2} s, {} Hz)", output.display(), audio.seconds(), audio.sample_rate);
                return Ok(());
            }
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
        Command::Render { project, output, format } => {
            #[derive(serde::Deserialize)]
            struct Project {
                sources: Vec<PathBuf>,
                #[serde(flatten)]
                arrangement: Arrangement,
            }
            let text = std::fs::read_to_string(&project).with_context(|| format!("reading {}", project.display()))?;
            let p: Project = serde_json::from_str(&text).with_context(|| format!("parsing {}", project.display()))?;
            let base = project.parent().unwrap_or(Path::new("."));
            let sources = p.sources.iter().map(|s| read(&base.join(s))).collect::<Result<Vec<_>>>()?;
            let audio = arrange::render(&p.arrangement, &sources)?;
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

fn find_genre(id: &str) -> Result<preset::Genre> {
    let genres = preset::builtin_genres();
    let ids: Vec<String> = genres.iter().map(|g| g.id.clone()).collect();
    genres
        .into_iter()
        .find(|g| g.id == id)
        .with_context(|| format!("no genre `{id}`; choose one of {}", ids.join(", ")))
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
