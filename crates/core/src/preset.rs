//! Genres and stations: plain TOML presets that shape every mode.
//!
//! A [`Genre`] is a system prompt for the model (a template that wraps what the listener types) plus defaults for
//! sampling and segment timing. A [`Station`] is radio mode's unit: a genre and a rotation of prompts. The presets that
//! ship with Driftscape live in `presets/` at the repository root and are compiled in; see [`builtin_genres`].

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Genre {
    pub id: String,
    pub name: String,
    pub description: String,
    /// The prompt sent to the model; `{prompt}` is replaced by the listener's words and `{bpm}` by the tempo.
    pub template: String,
    /// A few words that carry this genre into a blend with another, e.g. "a soft deep kick pulse".
    pub flavor: String,
    /// What plays when the listener has not typed anything.
    pub default_prompt: String,
    pub guidance: f32,
    pub temperature: f32,
    /// New audio per segment, in seconds.
    pub segment_seconds: f32,
    /// How much of the previous segment each new one replays to continue from it.
    pub overlap_seconds: f32,
    /// Crossfade between segments, inside the overlap.
    pub crossfade_seconds: f32,
    /// A fixed tempo for the whole session, for genres with a beat.
    pub bpm: Option<f32>,
}

impl Genre {
    /// The full prompt for the model.
    pub fn prompt(&self, listener: &str) -> String {
        let listener = if listener.trim().is_empty() { &self.default_prompt } else { listener.trim() };
        let bpm = self.bpm.map_or_else(String::new, |b| format!("{b:.0}"));
        self.template.replace("{prompt}", listener).replace("{bpm}", &bpm)
    }

    /// Frames per bar of 4/4 at `sample_rate`, for genres with a tempo.
    pub fn bar_frames(&self, sample_rate: u32) -> Option<usize> {
        self.bpm.map(|b| (4.0 * 60.0 / f64::from(b) * f64::from(sample_rate)).round() as usize)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Station {
    pub id: String,
    pub name: String,
    pub description: String,
    /// The [`Genre::id`] it plays in.
    pub genre: String,
    /// Prompts in rotation.
    pub prompts: Vec<String>,
    /// Segments before moving on to the next prompt.
    pub segments_per_prompt: u32,
}

impl Station {
    /// The prompt for the `n`-th segment of a session.
    pub fn prompt_for(&self, segment: u64) -> &str {
        if self.prompts.is_empty() {
            return "";
        }
        let per = u64::from(self.segments_per_prompt.max(1));
        &self.prompts[((segment / per) % self.prompts.len() as u64) as usize]
    }
}

pub fn parse_genre(toml: &str) -> Result<Genre, toml::de::Error> {
    toml::from_str(toml)
}

pub fn parse_station(toml: &str) -> Result<Station, toml::de::Error> {
    toml::from_str(toml)
}

macro_rules! presets {
    ($dir:literal: $($name:literal),+ $(,)?) => {
        [$((concat!($name), include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../presets/", $dir, "/", $name, ".toml")))),+]
    };
}

const GENRES: [(&str, &str); 5] =
    presets!("genres": "ambient", "synthetic-ambience", "realistic-ambience", "ambient-techno", "thematic");
const STATIONS: [(&str, &str); 5] =
    presets!("stations": "harbour-dusk", "deep-space", "night-drive", "overworld", "tavern");

/// The genres that ship with Driftscape. Panics only if a shipped file is malformed, which the tests rule out.
pub fn builtin_genres() -> Vec<Genre> {
    GENRES
        .iter()
        .map(|(name, src)| parse_genre(src).unwrap_or_else(|e| panic!("presets/genres/{name}.toml: {e}")))
        .collect()
}

/// The stations that ship with Driftscape.
pub fn builtin_stations() -> Vec<Station> {
    STATIONS
        .iter()
        .map(|(name, src)| parse_station(src).unwrap_or_else(|e| panic!("presets/stations/{name}.toml: {e}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_presets_are_consistent() {
        let genres = builtin_genres();
        for (g, (file, _)) in genres.iter().zip(GENRES) {
            assert_eq!(g.id, file, "file name and id differ");
            assert!(g.template.contains("{prompt}"), "{} has no {{prompt}}", g.id);
            assert!(!g.flavor.is_empty() && g.flavor.len() < 60, "{}: flavor should be a few words", g.id);
            assert_eq!(g.template.contains("{bpm}"), g.bpm.is_some(), "{}: {{bpm}} and bpm go together", g.id);
            // The crossfade sits at the end of the overlap; keep it half a second clear of the continued clip's start,
            // where the audio decoder has the least context.
            assert!(g.crossfade_seconds + 0.5 <= g.overlap_seconds, "{}: the crossfade must fit in the overlap", g.id);
            assert!(g.segment_seconds + g.overlap_seconds <= 30.0, "{}: MusicGen makes at most 30 s", g.id);
        }
        for (s, (file, _)) in builtin_stations().iter().zip(STATIONS) {
            assert_eq!(s.id, file);
            assert!(genres.iter().any(|g| g.id == s.genre), "{} plays unknown genre {}", s.id, s.genre);
            assert!(!s.prompts.is_empty());
        }
    }

    #[test]
    fn prompt_fills_the_template() {
        let g = builtin_genres().into_iter().find(|g| g.id == "ambient-techno").unwrap();
        let p = g.prompt("  rainy tram ride ");
        assert!(p.starts_with("rainy tram ride. "));
        assert!(p.contains("118 BPM"));
        assert!(g.prompt("").starts_with(&g.default_prompt));
    }

    #[test]
    fn stations_rotate_prompts() {
        let s = builtin_stations().into_iter().find(|s| s.id == "harbour-dusk").unwrap();
        assert_eq!(s.prompt_for(0), s.prompts[0]);
        assert_eq!(s.prompt_for(2), s.prompts[0]);
        assert_eq!(s.prompt_for(3), s.prompts[1]);
        assert_eq!(s.prompt_for(12), s.prompts[0]);
    }

    #[test]
    fn bars_at_tempo() {
        let g = builtin_genres().into_iter().find(|g| g.id == "ambient-techno").unwrap();
        // 4 beats at 118 BPM is 2.0339 s.
        assert_eq!(g.bar_frames(48_000), Some(97_627));
    }
}
