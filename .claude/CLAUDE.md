# Driftscape

A local AI music engine for ambient soundscapes. Inference always runs on the user's own machine. One Rust core,
several ways in: a CLI (primary), a DAW plugin, a local web GUI (`driftscape serve`) and a hosted static PWA.
The plan, with diagrams, modes and genres: `.claude/VISION.md`.

The Phoenix LiveView prototype lives on the `platform` branch.

## Layout

```
crates/core/   pure audio work: WAV, resampling, tempo, WSOLA time-stretch, seamless loops, tempo-fitted loops
               (fit_loop) and rejoin points, the real-time mixer and
               leveler, genre/station presets. No I/O, no model code, no threads.
crates/infer/  voices. MusicGen: text encoder + EnCodec on ONNX Runtime (q4 files from Xenova/musicgen-small),
               decoder on candle/Metal (gpu.rs, facebook/musicgen-small safetensors) or ONNX on CPU (onnx.rs)
crates/cli/    the `driftscape` binary: live, radio, gen, loop, play, info. live.rs runs live and radio mode
               (generator thread, control loop, audio callback, recorder thread).
crates/wasm/   C ABI over core for the browser (rejoin, tempo); no wasm-bindgen
presets/       genres/*.toml and stations/*.toml, compiled into core (preset.rs lists them)
web/           SvelteKit app (adapter-static, Svelte 5, #lib/* imports). lib/engine.js ports live.rs to Web Audio;
               lib/voice/ has the in-browser MusicGen worker (onnxruntime-web, WebGPU), the T5 tokenizer, and
               NativeVoice for `driftscape serve` (crates/cli/src/serve.rs embeds web/build)
```

Models live in `~/Library/Application Support/Driftscape/models/musicgen-small/` (`onnx/`, `tokenizer.json`,
`safetensors/model.safetensors`).

Planned: `crates/plugin` (nih-plug CLAP, wrapped to VST3/AU with clap-wrapper), `web/` (SvelteKit, adapter-static).

## Live mode

- Each segment continues from the previous one's last `overlap_seconds` of tokens, so its first frames reproduce
  the previous tail (measured: 0.99 correlation, zero lag; `--example continuation`). The crossfade sits in that overlap.
- If the next segment is late, the control loop crossfades back to the current segment's `rejoin` point (bar-aligned
  for genres with `bpm`). The next segment still continues from the buffer's end, which comes round again.
- The audio callback only applies commands and mixes; buffers are freed on the control thread.
- A tweak bumps a version counter: the waiting segment is dropped, the one being generated is cancelled (via the
  progress callback), and the next is made from the *playing* segment's tokens. It is short (6 s), marked urgent, and
  crossfades in (3 s) as soon as it's ready. Measured delay: 5–10 s.
- Segments stay full length otherwise: each replays the overlap first, so short segments lose throughput. The overlap
  is fed as one batch (`Decoder::prefill`; GPU: 150 positions in ~160 ms instead of ~1.35 s; `--example prefill`
  checks it matches stepping).
- The control loop warns if the audio clock stops (device change). `:quit` and Ctrl-C never wait on a lock; a second
  Ctrl-C exits at once.
- Open: one session (2026-11-21) went silent after a urgent transition and ignored `:quit` for ~55 min. Not
  reproduced since; the hardening above addresses the plausible causes. `DRIFTSCAPE_TRACE=1` prints a clock heartbeat.
- Tempo: MusicGen follows a BPM in the prompt closely (118 asked → 118.0 measured, every segment of a 2.5 min
  night-drive session, and 118.01–118.03 in `gen`). Live mode therefore does not stretch segments (crossfading two
  stretched copies would flam); `fit_loop`/`stretch` are for fixed outside tempos: `gen --bpm --bars`, `loop
  --to-bpm`, and later the plugin following the host.
- Adding a preset: drop a TOML file in `presets/` and list it in `crates/core/src/preset.rs`; the tests check it.

## Web notes

- `web/scripts/prepare-assets.sh` builds the core wasm, exports presets (`driftscape presets --json`), and ships
  onnxruntime-web's 28 MB WebGPU runtime gzipped (Cloudflare Pages caps files at 25 MiB). The worker sniffs the gzip
  magic because some servers send `.gz` with `Content-Encoding: gzip` and the browser unpacks it first.
- Vite resolves `onnxruntime-web` with the `onnxruntime-web-use-extern-wasm` condition; the default build makes Vite
  copy the 28 MB wasm into the output.
- onnxruntime-web refuses to create WebGPU sessions concurrently: create them one by one.
- The tokenizer is checked against the Rust `tokenizers` crate (`npm test`; fixtures from `--example tokenize`).
- `vite preview` does not pick up a rebuild; restart it. Unregister the service worker when testing a new build.
- The app picks NativeVoice when `/api/health` answers `{"engine":"driftscape"}`, else WorkerVoice.

## Rules

- Logic goes in `crates/core`. It stays target-independent so it runs natively and as wasm: no `std::fs`, `std::net`,
  threads, `wasm_bindgen` or `web_sys` there.
- Audio is planar `f32` (`driftscape_core::Audio`). MusicGen outputs 32 kHz mono.
- Every DSP change gets a test with a synthetic signal that shows the property (no click at the seam, tempo within
  0.1 BPM, no aliasing).
- Never block an audio callback: resample and prepare buffers before the stream starts; the callback only copies.

## Inference notes

- `cargo run --release -p driftscape-infer --features metal --example parity -- <model dir>` compares the GPU decoder's
  logits with the ONNX one. Step 0 must correlate at 1.00; later steps ~0.93–0.98 (q4 noise). Run it after any change
  to `gpu.rs`.
- `--example stepcost <model dir> [position]` splits a GPU step into CPU queueing and GPU time at a sequence
  position. Real time needs < 20 ms per step; currently ~9 ms at 20, ~10.5 ms at 400, ~12 ms at 1000 (M4, q8_0).
  Check a late position too: per-step costs that grow with the sequence hide in early-position measurements.
- Env knobs: `DRIFTSCAPE_TRACE=1` (stage timings), `DRIFTSCAPE_QUANT=q8_0|q4k|none` (GPU weight quantization, default q8_0; q4k loses accuracy),
  `DRIFTSCAPE_DTYPE=f16|bf16` (GPU precision when QUANT=none),
  `DRIFTSCAPE_THREADS=n` (ONNX CPU threads).
- `crates/infer/build.rs` links clang's runtime: ONNX Runtime's CoreML code needs `__isPlatformVersionAtLeast`.

## Commands

```bash
mise run test                     # cargo test, clippy -D warnings, fmt --check
cargo run -p driftscape -- --help
```
