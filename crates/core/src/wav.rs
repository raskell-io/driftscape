//! WAV reading and writing on byte slices, so the core never touches the file system.
//!
//! Reads PCM (8, 16, 24 and 32 bit) and IEEE float (32 and 64 bit), including `WAVE_FORMAT_EXTENSIBLE`. Writes 16-
//! or 24-bit PCM or 32-bit float.

use crate::Audio;
use std::fmt;

/// The sample format to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SampleFormat {
    Pcm16,
    #[default]
    Pcm24,
    Float32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WavError {
    NotRiffWave,
    MissingChunk(&'static str),
    Unsupported(String),
    Malformed(&'static str),
}

impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotRiffWave => write!(f, "not a RIFF/WAVE file"),
            Self::MissingChunk(c) => write!(f, "WAV file has no `{c}` chunk"),
            Self::Unsupported(what) => write!(f, "unsupported WAV format: {what}"),
            Self::Malformed(what) => write!(f, "malformed WAV file: {what}"),
        }
    }
}

impl std::error::Error for WavError {}

const PCM: u16 = 1;
const FLOAT: u16 = 3;
const EXTENSIBLE: u16 = 0xFFFE;

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// Parse a WAV file into planar `f32` audio.
pub fn decode(bytes: &[u8]) -> Result<Audio, WavError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(WavError::NotRiffWave);
    }

    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, rate, bits)
    let mut data: Option<&[u8]> = None;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32_at(bytes, pos + 4) as usize;
        let body_start = pos + 8;
        // Writers that stream sometimes leave the data size at 0 or too large; take what is there.
        let body_end = body_start.saturating_add(size).min(bytes.len());
        let body = &bytes[body_start..body_end];
        match id {
            b"fmt " => {
                if body.len() < 16 {
                    return Err(WavError::Malformed("fmt chunk too short"));
                }
                let mut format = u16_at(body, 0);
                if format == EXTENSIBLE {
                    if body.len() < 26 {
                        return Err(WavError::Malformed("extensible fmt chunk too short"));
                    }
                    format = u16_at(body, 24); // first two bytes of the sub-format GUID
                }
                fmt = Some((format, u16_at(body, 2), u32_at(body, 4), u16_at(body, 14)));
            }
            b"data" => {
                let body = if size == 0 { &bytes[body_start..] } else { body };
                data = Some(body);
            }
            _ => {}
        }
        // Chunks are padded to an even length.
        pos = body_start.saturating_add(size + (size & 1));
    }

    let (format, channels, rate, bits) = fmt.ok_or(WavError::MissingChunk("fmt "))?;
    let data = data.ok_or(WavError::MissingChunk("data"))?;
    if channels == 0 || rate == 0 {
        return Err(WavError::Malformed("zero channels or sample rate"));
    }

    let read: fn(&[u8]) -> f32 = match (format, bits) {
        (PCM, 8) => |s| (f32::from(s[0]) - 128.0) / 128.0,
        (PCM, 16) => |s| f32::from(i16::from_le_bytes([s[0], s[1]])) / 32768.0,
        (PCM, 24) => |s| (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8_388_608.0,
        (PCM, 32) => |s| i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0,
        (FLOAT, 32) => |s| f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        (FLOAT, 64) => |s| f64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]) as f32,
        (f, b) => return Err(WavError::Unsupported(format!("format tag {f}, {b} bits"))),
    };

    let width = usize::from(bits / 8);
    let channels = usize::from(channels);
    let frame = width * channels;
    let frames = data.len() / frame;
    let mut out = vec![Vec::with_capacity(frames); channels];
    for f in 0..frames {
        for (c, ch) in out.iter_mut().enumerate() {
            let i = f * frame + c * width;
            ch.push(read(&data[i..i + width]));
        }
    }
    Ok(Audio::new(rate, out))
}

/// Write audio as a WAV file. Samples outside `-1.0..=1.0` are clipped for the PCM formats.
pub fn encode(audio: &Audio, format: SampleFormat) -> Vec<u8> {
    let (tag, bits): (u16, u16) = match format {
        SampleFormat::Pcm16 => (PCM, 16),
        SampleFormat::Pcm24 => (PCM, 24),
        SampleFormat::Float32 => (FLOAT, 32),
    };
    let channels = audio.channel_count() as u16;
    let block_align = channels * bits / 8;
    let data_len = audio.len() as u32 * u32::from(block_align);

    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&tag.to_le_bytes());
    b.extend_from_slice(&channels.to_le_bytes());
    b.extend_from_slice(&audio.sample_rate.to_le_bytes());
    b.extend_from_slice(&(audio.sample_rate * u32::from(block_align)).to_le_bytes());
    b.extend_from_slice(&block_align.to_le_bytes());
    b.extend_from_slice(&bits.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());

    for i in 0..audio.len() {
        for ch in &audio.channels {
            let s = ch[i];
            match format {
                SampleFormat::Pcm16 => {
                    let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                    b.extend_from_slice(&v.to_le_bytes());
                }
                SampleFormat::Pcm24 => {
                    let v = (s.clamp(-1.0, 1.0) * 8_388_607.0).round() as i32;
                    b.extend_from_slice(&v.to_le_bytes()[..3]);
                }
                SampleFormat::Float32 => b.extend_from_slice(&s.to_le_bytes()),
            }
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_audio() -> Audio {
        Audio::new(32_000, vec![vec![0.0, 0.5, -0.5, 0.999], vec![0.25, -0.25, 0.125, -1.0]])
    }

    #[test]
    fn roundtrips_every_format() {
        let a = sample_audio();
        for (format, tolerance) in
            [(SampleFormat::Pcm16, 2.0 / 32_768.0), (SampleFormat::Pcm24, 1e-6), (SampleFormat::Float32, 0.0)]
        {
            let back = decode(&encode(&a, format)).unwrap();
            assert_eq!(back.sample_rate, 32_000);
            assert_eq!(back.channel_count(), 2);
            for (x, y) in a.channels.iter().flatten().zip(back.channels.iter().flatten()) {
                assert!((x - y).abs() <= tolerance, "{format:?}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn skips_unknown_chunks() {
        let mut bytes = encode(&sample_audio(), SampleFormat::Pcm16);
        // Insert a LIST chunk with an odd length (padded) between fmt and data.
        let list = [b"LIST".as_slice(), &3u32.to_le_bytes(), b"abc\0"].concat();
        bytes.splice(36..36, list);
        let back = decode(&bytes).unwrap();
        assert_eq!(back.len(), 4);
    }

    #[test]
    fn rejects_non_wav() {
        assert_eq!(decode(b"hello world, not a wav"), Err(WavError::NotRiffWave));
    }
}
