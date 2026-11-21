//! Feed the CPU (ONNX q4) and GPU (candle) decoders the same text states and tokens and compare their logits.
//! The ONNX weights are 4-bit quantized, so expect close agreement, not equality.
use driftscape_infer::musicgen::{Decoder, gpu::GpuDecoder, onnx::OnnxDecoder};
use ort::value::Tensor;

fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("model directory"));
    let env = ort::environment::init().with_telemetry(false).build()?;
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids: Vec<i64> =
        tokenizer.encode("warm ambient pads", true).unwrap().get_ids().iter().map(|&i| i64::from(i)).collect();
    let seq = ids.len();
    let mut text = ort::session::Session::builder(&env)?.commit_from_file(dir.join("onnx/text_encoder_q4.onnx"))?;
    let out = text.run(ort::inputs![
        "input_ids" => Tensor::from_array(([1, seq], ids))?,
        "attention_mask" => Tensor::from_array(([1, seq], vec![1i64; seq]))?,
    ])?;
    let mut states = out["last_hidden_state"].try_extract_tensor::<f32>()?.1.to_vec();
    states.extend(std::iter::repeat_n(0.0, seq * 768));
    let mut mask = vec![1i64; seq];
    mask.extend(std::iter::repeat_n(0, seq));

    let mut cpu = OnnxDecoder::load(&env, &dir.join("onnx/decoder_model_merged_q4.onnx"))?;
    let mut gpu = GpuDecoder::load(&dir.join("safetensors/model.safetensors"))?;
    cpu.start(&states, &mask, seq)?;
    gpu.start(&states, &mask, seq)?;

    let mut tokens = vec![2048i64; 8];
    let (mut agree, mut total, mut worst_corr) = (0, 0, 1.0f64);
    let mut by_row = [(0.0f64, 0usize); 8];
    let mut by_step = Vec::new();
    for step in 0..40 {
        let (a, b) = (cpu.step(&tokens)?, gpu.step(&tokens)?);
        for row in 0..8 {
            let (x, y) = (&a[row * 2048..][..2048], &b[row * 2048..][..2048]);
            let argmax = |v: &[f32]| v.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).unwrap().0;
            agree += usize::from(argmax(x) == argmax(y));
            total += 1;
            let mean = |v: &[f32]| v.iter().map(|&f| f64::from(f)).sum::<f64>() / v.len() as f64;
            let (mx, my) = (mean(x), mean(y));
            let cov: f64 = x.iter().zip(y).map(|(&p, &q)| (f64::from(p) - mx) * (f64::from(q) - my)).sum();
            let vx: f64 = x.iter().map(|&p| (f64::from(p) - mx).powi(2)).sum();
            let vy: f64 = y.iter().map(|&q| (f64::from(q) - my).powi(2)).sum();
            let c = cov / (vx * vy).sqrt();
            worst_corr = worst_corr.min(c);
            by_row[row].0 += c;
            by_row[row].1 += 1;
            if row == 0 {
                by_step.push(c);
            }
        }
        // Continue with the CPU decoder's greedy choice, same tokens into both; pad the delayed codebooks early on.
        for k in 0..4 {
            let row = &a[k * 2048..][..2048];
            let t = if step < k {
                2048
            } else {
                row.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).unwrap().0 as i64
            };
            tokens[k] = t;
            tokens[4 + k] = t;
        }
    }
    println!(
        "mean correlation by row (cond 0-3, uncond 4-7): {:?}",
        by_row.iter().map(|(s, n)| format!("{:.3}", s / *n as f64)).collect::<Vec<_>>()
    );
    println!("row 0 by step: {:?}", by_step.iter().map(|c| format!("{c:.2}")).collect::<Vec<_>>());
    println!("top-1 agreement {agree}/{total}, worst logit correlation {worst_corr:.4}");
    Ok(())
}
