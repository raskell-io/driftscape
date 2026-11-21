//! Generate a clip, continue it from its last 3 s, and measure how closely the continuation's start reproduces the
//! first clip's end (the region live mode crossfades over).
use driftscape_infer::{
    Request, Voice,
    musicgen::{Backend, MusicGen},
};

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt().max(1e-12)
}

fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("model directory"));
    let env = ort::environment::init().with_telemetry(false).build()?;
    let mut voice = MusicGen::load(&env, &dir, Backend::Gpu)?;
    let prompt = "slow warm ambient pads, soft piano";
    let a = voice.generate(&Request::new(prompt, 8.0, 1), &mut |_| driftscape_infer::ControlFlow::Continue(()))?;
    let b = voice
        .generate(&Request { continue_from: Some(a.tokens.tail(3.0)), ..Request::new(prompt, 5.0, 2) }, &mut |_| {
            driftscape_infer::ControlFlow::Continue(())
        })?;
    let sr = 32_000usize;
    let overlap = 3 * sr;
    println!("a {:.2} s, b {:.2} s, b overlap {:.2} s", a.audio.seconds(), b.audio.seconds(), b.overlap_seconds);
    let (a, b) = (&a.audio.channels[0], &b.audio.channels[0]);
    let a_tail = &a[a.len() - overlap..];
    // Whole overlap, then per half second, then the best lag within ±20 ms.
    println!("overlap correlation {:.3}", correlation(a_tail, &b[..overlap]));
    for i in 0..6 {
        let r = i * sr / 2..(i + 1) * sr / 2;
        print!("{:.2} ", correlation(&a_tail[r.clone()], &b[r]));
    }
    println!("(per 0.5 s)");
    let mid = sr..2 * sr;
    let best = (-640i64..=640)
        .map(|lag| {
            let s = (mid.start as i64 + lag) as usize;
            (lag, correlation(&a_tail[mid.clone()], &b[s..s + sr]))
        })
        .max_by(|x, y| x.1.total_cmp(&y.1))
        .unwrap();
    println!("best lag {} samples, correlation {:.3}", best.0, best.1);
    Ok(())
}
