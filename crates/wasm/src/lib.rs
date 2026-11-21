//! The WebAssembly boundary: a C ABI over `driftscape-core`, nothing else. JavaScript asks for a sample buffer, copies
//! audio into the module's linear memory, and calls the analysis functions on it. No wasm-bindgen.
//!
//! All functions are single-threaded and work on the one buffer from the last [`buffer`] call.

use driftscape_core::{looping::rejoin_point, tempo};

static mut SAMPLES: Vec<f32> = Vec::new();

/// Make room for `len` mono samples and return where they start in linear memory. JavaScript writes the audio there.
#[unsafe(no_mangle)]
pub extern "C" fn buffer(len: u32) -> *mut f32 {
    // SAFETY: the module is single-threaded and this is the only place SAMPLES is resized.
    #[allow(static_mut_refs)]
    unsafe {
        SAMPLES.clear();
        SAMPLES.resize(len as usize, 0.0);
        SAMPLES.as_mut_ptr()
    }
}

fn samples() -> &'static [f32] {
    // SAFETY: as above; only read between `buffer` calls.
    #[allow(static_mut_refs)]
    unsafe {
        &SAMPLES
    }
}

/// Where to jump back to when the buffer has to repeat (see `driftscape_core::looping::rejoin_point`). `bar` of 0
/// means no tempo. Returns the frame, or -1 when no candidate fits or the arguments are out of range.
#[unsafe(no_mangle)]
pub extern "C" fn rejoin(from: u32, window: u32, earliest: u32, latest: u32, bar: u32) -> i64 {
    let s = samples();
    let (from, window) = (from as usize, window as usize);
    if window == 0 || from + window > s.len() {
        return -1;
    }
    let bar = (bar > 0).then_some(bar as usize);
    rejoin_point(s, from, window, earliest as usize, latest as usize, bar).map_or(-1, |(f, _)| f as i64)
}

/// The buffer's tempo in BPM at `sample_rate`, or 0 when it has no steady pulse.
#[unsafe(no_mangle)]
pub extern "C" fn tempo(sample_rate: u32) -> f64 {
    if sample_rate == 0 {
        return 0.0;
    }
    tempo::estimate(samples(), sample_rate, tempo::Range::default()).map_or(0.0, |t| t.bpm)
}
