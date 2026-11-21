# Driftscape

A local AI music engine for ambient soundscapes. Inference always runs on the user's own machine. One Rust core,
several ways in: a CLI (primary), a DAW plugin, a local web GUI (`driftscape serve`) and a hosted static PWA.
The plan, with diagrams, modes and genres: `.claude/VISION.md`.

The Phoenix LiveView prototype lives on the `platform` branch.

## Layout

```
crates/core/   pure audio work: WAV, resampling, tempo, seamless loops. No I/O, no model code, no threads.
crates/cli/    the `driftscape` binary (loop, play, info so far)
```

Planned: `crates/infer` (voice trait, MusicGen via ort), `crates/wasm` (C ABI over core, no wasm-bindgen),
`crates/plugin` (nih-plug CLAP, wrapped to VST3/AU with clap-wrapper), `web/` (SvelteKit, adapter-static).

## Rules

- Logic goes in `crates/core`. It stays target-independent so it runs natively and as wasm: no `std::fs`, `std::net`,
  threads, `wasm_bindgen` or `web_sys` there.
- Audio is planar `f32` (`driftscape_core::Audio`). MusicGen outputs 32 kHz mono.
- Every DSP change gets a test with a synthetic signal that shows the property (no click at the seam, tempo within
  0.1 BPM, no aliasing).
- Never block an audio callback: resample and prepare buffers before the stream starts; the callback only copies.

## Commands

```bash
mise run test                     # cargo test, clippy -D warnings, fmt --check
cargo run -p driftscape -- --help
```
