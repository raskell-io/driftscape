//! Looping playback on the default output device.

use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use driftscape_core::{Audio, resample::resample};

/// Play `audio` on repeat until the process is stopped.
pub fn looped(audio: &Audio) -> Result<()> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or_else(|| anyhow!("no audio output device"))?;
    let config = device.default_output_config().context("querying the output device")?;
    if config.sample_format() != cpal::SampleFormat::F32 {
        return Err(anyhow!("output device wants {:?} samples; only f32 is supported so far", config.sample_format()));
    }
    let config: cpal::StreamConfig = config.into();
    let device_channels = usize::from(config.channels);
    let device_rate = config.sample_rate;

    // Resample once up front so the audio callback only copies.
    let audio = resample(audio, device_rate);
    let source_channels = audio.channel_count();
    let len = audio.len();
    let mut pos = 0usize;

    let stream = device
        .build_output_stream(
            config,
            move |out: &mut [f32], _| {
                for frame in out.chunks_mut(device_channels) {
                    for (c, sample) in frame.iter_mut().enumerate() {
                        // Mono goes to every output channel; extra source channels beyond the device's are dropped.
                        *sample = audio.channels[c.min(source_channels - 1)][pos];
                    }
                    pos = (pos + 1) % len;
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .context("opening the audio stream")?;
    stream.play().context("starting playback")?;
    loop {
        std::thread::park();
    }
}
