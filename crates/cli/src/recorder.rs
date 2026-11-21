//! Streaming WAV recording: samples are appended as they arrive and the header is completed when the recording ends,
//! so a session of any length never sits in memory.

use anyhow::{Context, Result};
use std::{
    fs::File,
    io::{BufWriter, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

/// 24-bit PCM, interleaved.
pub struct WavRecorder {
    out: BufWriter<File>,
    path: PathBuf,
    channels: u16,
    /// Samples written (all channels).
    samples: u32,
}

impl WavRecorder {
    pub fn create(path: &Path, sample_rate: u32, channels: u16) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut out = BufWriter::new(file);
        let bits = 24u16;
        let block_align = channels * bits / 8;
        out.write_all(b"RIFF\0\0\0\0WAVEfmt ")?;
        out.write_all(&16u32.to_le_bytes())?;
        out.write_all(&1u16.to_le_bytes())?; // PCM
        out.write_all(&channels.to_le_bytes())?;
        out.write_all(&sample_rate.to_le_bytes())?;
        out.write_all(&(sample_rate * u32::from(block_align)).to_le_bytes())?;
        out.write_all(&block_align.to_le_bytes())?;
        out.write_all(&bits.to_le_bytes())?;
        out.write_all(b"data\0\0\0\0")?;
        Ok(Self { out, path: path.to_owned(), channels, samples: 0 })
    }

    /// Append interleaved samples.
    pub fn write(&mut self, samples: &[f32]) -> Result<()> {
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * 8_388_607.0).round() as i32;
            self.out.write_all(&v.to_le_bytes()[..3])?;
        }
        self.samples += samples.len() as u32;
        Ok(())
    }

    pub fn seconds(&self, sample_rate: u32) -> f64 {
        f64::from(self.samples / u32::from(self.channels)) / f64::from(sample_rate)
    }

    /// Fill in the sizes and close the file.
    pub fn finish(mut self) -> Result<PathBuf> {
        let data = self.samples * 3;
        self.out.seek(SeekFrom::Start(4))?;
        self.out.write_all(&(36 + data).to_le_bytes())?;
        self.out.seek(SeekFrom::Start(40))?;
        self.out.write_all(&data.to_le_bytes())?;
        self.out.flush()?;
        Ok(self.path)
    }
}

/// `YYYYMMDD-HHMMSS` in UTC, for file names, without a date library.
pub fn timestamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}{month:02}{day:02}-{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_is_a_valid_wav() {
        let path = std::env::temp_dir().join(format!("driftscape-rec-test-{}.wav", std::process::id()));
        let mut r = WavRecorder::create(&path, 48_000, 2).unwrap();
        r.write(&[0.0, 0.5, -0.5]).unwrap();
        r.write(&[0.25]).unwrap();
        assert_eq!(r.seconds(48_000), 2.0 / 48_000.0);
        let path = r.finish().unwrap();
        let audio = driftscape_core::wav::decode(&std::fs::read(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(audio.sample_rate, 48_000);
        assert_eq!(audio.channel_count(), 2);
        assert_eq!(audio.len(), 2);
        assert!((audio.channels[1][0] - 0.5).abs() < 1e-6);
        assert!((audio.channels[1][1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn timestamp_has_the_right_shape() {
        let t = timestamp();
        assert_eq!(t.len(), 15);
        assert!(t.starts_with("20"));
        assert_eq!(&t[8..9], "-");
    }
}
