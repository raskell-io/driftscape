//! Check that feeding positions as one batch (prefill) gives the same logits as stepping through them one by one.
use driftscape_infer::musicgen::{Decoder, gpu::GpuDecoder};

fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("model directory"));
    let mut gpu = GpuDecoder::load(&dir.join("safetensors/model.safetensors"))?;
    let seq = 6;
    let states: Vec<f32> = (0..2 * seq * 768).map(|i| ((i * 7919) % 1000) as f32 / 5000.0 - 0.1).collect();
    let mask: Vec<i64> = (0..2 * seq).map(|i| i64::from(i < seq)).collect();
    let mut x = 12345u64;
    let mut token = || {
        x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1442695040888963407);
        ((x >> 33) % 2048) as i64
    };
    for n in [2usize, 40, 150] {
        let positions: Vec<Vec<i64>> = (0..n)
            .map(|_| {
                let t: Vec<i64> = (0..4).map(|_| token()).collect();
                t.iter().chain(t.iter()).copied().collect()
            })
            .collect();
        gpu.start(&states, &mask, seq)?;
        let mut stepped = Vec::new();
        let t = std::time::Instant::now();
        for p in &positions {
            stepped = gpu.step(p)?;
        }
        let step_time = t.elapsed();
        gpu.start(&states, &mask, seq)?;
        let t = std::time::Instant::now();
        let batched = gpu.prefill(&positions)?;
        let batch_time = t.elapsed();
        let max_diff = stepped.iter().zip(&batched).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let same_top =
            (0..8).filter(|r| argmax(&stepped[r * 2048..][..2048]) == argmax(&batched[r * 2048..][..2048])).count();
        println!(
            "{n:>3} positions: max |diff| {max_diff:.4}, same top token {same_top}/8, stepped {:.0} ms, batched {:.0} ms",
            step_time.as_secs_f64() * 1e3,
            batch_time.as_secs_f64() * 1e3
        );
    }
    Ok(())
}
