//! Print the built-in genres and stations as JSON, for the web app (web/scripts/prepare-assets.sh). Lives in the core
//! so the web build needs nothing heavier than this crate (no ONNX Runtime, no audio output).
use driftscape_core::preset::{builtin_genres, builtin_stations};

fn main() {
    let json = serde_json::json!({ "genres": builtin_genres(), "stations": builtin_stations() });
    println!("{}", serde_json::to_string_pretty(&json).expect("presets serialize"));
}
