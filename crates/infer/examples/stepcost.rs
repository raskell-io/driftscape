//! Split a GPU decoding step into time spent queueing work on the CPU and time waiting for the GPU.
use driftscape_infer::musicgen::{Decoder, gpu::GpuDecoder};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("model directory"));
    let mut gpu = GpuDecoder::load(&dir.join("safetensors/model.safetensors"))?;
    let seq = 8;
    gpu.start(&vec![0.01; 2 * seq * 768], &[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], seq)?;
    let tokens = vec![7i64; 8];
    for _ in 0..20 {
        gpu.step(&tokens)?; // warm up kernels
    }
    let (mut queue, mut total) = (0.0, 0.0);
    let n = 100;
    for _ in 0..n {
        let t = Instant::now();
        let pending = gpu.step_lazy(&tokens)?;
        queue += t.elapsed().as_secs_f64();
        let _ = pending.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
        total += t.elapsed().as_secs_f64();
    }
    println!("per step: {:.1} ms total, {:.1} ms queueing on the CPU", total / n as f64 * 1e3, queue / n as f64 * 1e3);
    Ok(())
}
