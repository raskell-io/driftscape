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
//! - [`stretch`]: change a clip's length (and tempo) without changing its pitch.
//! - [`looping`]: cut a clip to whole bars and make the wrap-around seamless; find where to rejoin a repeat.
//! - [`mixer`]: the real-time mixer live mode plays through.
//! - [`preset`]: genres and stations.
//! - [`arrange`]: producer mode's timeline of layered clips, and its mixdown.

pub mod arrange;
pub mod audio;
pub mod looping;
pub mod mixer;
pub mod preset;
pub mod resample;
pub mod stretch;
pub mod tempo;
pub mod wav;

pub use audio::Audio;
