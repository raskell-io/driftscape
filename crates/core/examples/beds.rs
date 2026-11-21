//! Levels of the ambience beds, and optionally WAV files to listen to:
//! cargo run --release -p driftscape-core --example beds [out-dir]
use driftscape_core::{
    ambience::{Kind, render},
    wav::{SampleFormat, encode},
};

fn main() {
    let dir = std::env::args().nth(1);
    for kind in Kind::ALL {
        for intensity in [0.2, 0.6, 1.0] {
            let a = render(kind, 48_000, 30.0, 1, intensity);
            let l = &a.channels[0];
            let rms = (l.iter().map(|x| x * x).sum::<f32>() / l.len() as f32).sqrt();
            println!(
                "{:6} intensity {intensity:.1}: rms {:6.1} dBFS, peak {:5.2}",
                kind.id(),
                20.0 * rms.log10(),
                a.peak()
            );
            if let (Some(dir), true) = (&dir, intensity == 0.6) {
                std::fs::write(format!("{dir}/{}.wav", kind.id()), encode(&a, SampleFormat::Pcm16)).unwrap();
            }
        }
    }
}
