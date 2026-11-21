//! A deterministic reference: greedy decoding (top-k 1) of a fixed prompt, printing the first frames of each codebook
//! and simple audio statistics. The web voice's self-test runs the same request; matching tokens mean the backend
//! computes the same model.
use driftscape_infer::{
    ControlFlow, Request, Voice,
    musicgen::{Backend, MusicGen},
};

fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("model directory"));
    let backend = if std::env::args().nth(2).as_deref() == Some("gpu") { Backend::Gpu } else { Backend::Cpu };
    let env = ort::environment::init().with_telemetry(false).build()?;
    let mut voice = MusicGen::load(&env, &dir, backend)?;
    let request = Request { top_k: 1, ..Request::new("warm ambient pads, soft piano", 1.0, 1) };
    let clip = voice.generate(&request, &mut |_| ControlFlow::Continue(()))?;
    for (k, c) in clip.tokens.codebooks.iter().enumerate() {
        println!("codebook {k}: {:?}", &c[..12]);
    }
    let x = &clip.audio.channels[0];
    let rms = (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt();
    let zcr = x.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count() as f32 / x.len() as f32;
    println!("rms {rms:.4}  zero-crossing rate {zcr:.4}");
    if std::env::args().nth(3).as_deref() == Some("--json") {
        let json = format!(
            "{{\"prompt\":{:?},\"codebooks\":{:?},\"rms\":{rms},\"zcr\":{zcr}}}",
            request.prompt, clip.tokens.codebooks
        );
        std::fs::write("web/src/lib/voice/selfcheck.reference.json", json)?;
        println!("wrote web/src/lib/voice/selfcheck.reference.json");
    }
    Ok(())
}
