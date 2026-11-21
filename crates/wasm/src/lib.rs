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

// Producer mode: the arrangement's mixdown (see `driftscape_core::arrange`). Sources stay loaded across renders, so an
// edit only resends the tracks: `arr_tracks_clear`, then `arr_track` and `arr_clip` calls, then `arr_render`.

use driftscape_core::{
    Audio,
    arrange::{Arrangement, Curve, Placed, Track, render},
};

struct Producer {
    sources: Vec<Audio>,
    arrangement: Arrangement,
    output: Option<Audio>,
}

static mut PRODUCER: Option<Producer> = None;

fn producer() -> &'static mut Producer {
    // SAFETY: single-threaded module; no reference outlives the exported call that took it.
    #[allow(static_mut_refs)]
    unsafe {
        PRODUCER.get_or_insert_with(|| Producer {
            sources: Vec::new(),
            arrangement: Arrangement::new(32000),
            output: None,
        })
    }
}

/// Drop every source and track, and set the mixdown's sample rate.
#[unsafe(no_mangle)]
pub extern "C" fn arr_reset(sample_rate: u32) {
    let p = producer();
    p.sources.clear();
    p.output = None;
    p.arrangement = Arrangement::new(sample_rate.max(1));
}

/// Change the mixdown's sample rate and keep the sources (export renders at a DAW's rate, the preview at the voice's).
#[unsafe(no_mangle)]
pub extern "C" fn arr_set_rate(sample_rate: u32) {
    let p = producer();
    p.arrangement.sample_rate = sample_rate.max(1);
    p.output = None;
}

/// Add a silent source of `frames` frames and `channels` channels (1 or 2) at `rate`; returns its index, or -1.
/// JavaScript then fills each channel through [`arr_channel`].
#[unsafe(no_mangle)]
pub extern "C" fn arr_source(frames: u32, channels: u32, rate: u32) -> i32 {
    if !(1..=2).contains(&channels) || rate == 0 {
        return -1;
    }
    let p = producer();
    p.sources.push(Audio::new(rate, vec![vec![0.0; frames as usize]; channels as usize]));
    p.sources.len() as i32 - 1
}

/// Where a source's channel starts in linear memory, or null.
#[unsafe(no_mangle)]
pub extern "C" fn arr_channel(source: u32, channel: u32) -> *mut f32 {
    producer()
        .sources
        .get_mut(source as usize)
        .and_then(|s| s.channels.get_mut(channel as usize))
        .map_or(std::ptr::null_mut(), |c| c.as_mut_ptr())
}

/// Remove every track (the sources stay).
#[unsafe(no_mangle)]
pub extern "C" fn arr_tracks_clear() {
    producer().arrangement.tracks.clear();
}

/// Start a new track; the following [`arr_clip`] calls add to it.
#[unsafe(no_mangle)]
pub extern "C" fn arr_track(gain_db: f32, pan: f32, mute: u32) {
    producer().arrangement.tracks.push(Track { name: String::new(), gain_db, pan, mute: mute != 0, clips: Vec::new() });
}

/// Add a clip to the last track. A negative `length` plays to the source's end. Returns 0, or -1 without a track.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn arr_clip(
    source: u32,
    at: f64,
    offset: f64,
    length: f64,
    fade_in: f64,
    fade_out: f64,
    linear: u32,
    gain_db: f32,
) -> i32 {
    let Some(track) = producer().arrangement.tracks.last_mut() else { return -1 };
    track.clips.push(Placed {
        source: source as usize,
        at,
        offset,
        length: (length >= 0.0).then_some(length),
        fade_in,
        fade_out,
        curve: if linear != 0 { Curve::Linear } else { Curve::EqualPower },
        gain_db,
    });
    0
}

/// Mix down. Returns the output's length in frames, or -1 if a clip names a missing source.
#[unsafe(no_mangle)]
pub extern "C" fn arr_render() -> i64 {
    let p = producer();
    match render(&p.arrangement, &p.sources) {
        Ok(out) => {
            let n = out.len() as i64;
            p.output = Some(out);
            n
        }
        Err(_) => {
            p.output = None;
            -1
        }
    }
}

/// The last mixdown's channel (0 left, 1 right), or null.
#[unsafe(no_mangle)]
pub extern "C" fn arr_output(channel: u32) -> *const f32 {
    producer().output.as_ref().and_then(|o| o.channels.get(channel as usize)).map_or(std::ptr::null(), |c| c.as_ptr())
}
