//! Driftscape's audio core.
//!
//! Everything here is pure computation on sample buffers: no files, no sockets, no threads, no model code. That keeps
//! it testable with plain `cargo test` and lets the same code run in the CLI, the DAW plugin and, compiled to wasm,
//! the browser.
//!
//! - [`Audio`]: a planar buffer of `f32` samples.
//! - [`wav`]: read and write WAV bytes.
//! - [`resample`]: change the sample rate with a windowed-sinc filter.
//! - [`tempo`]: estimate the tempo of a clip.
//! - [`looping`]: cut a clip to whole bars and make the wrap-around seamless; find where to rejoin a repeat.
//! - [`mixer`]: the real-time mixer live mode plays through.
//! - [`preset`]: genres and stations.

pub mod audio;
pub mod looping;
pub mod mixer;
pub mod preset;
pub mod resample;
pub mod tempo;
pub mod wav;

pub use audio::Audio;
