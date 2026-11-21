# Driftscape — Vision

Re-imagined on 2026-11-21. The first version (February 2026) was a SoundCloud-style platform for ambient music; that
Phoenix prototype lives on the `platform` branch. Driftscape is now an engine first.

## What it is

A local AI music engine for ambient soundscapes. Inference always runs on the listener's own machine, so it costs us
nothing to run and works offline. One Rust core, four ways in:

1. **CLI** (primary): `driftscape live`, `radio`, `gen`, `loop`, `play`, `serve`.
2. **DAW plugin**: Logic Pro and Ableton Live. nih-plug CLAP, wrapped to VST3 and AUv2 with clap-wrapper.
3. **Local web GUI**: `driftscape serve` embeds the web app and gives it the native engine.
4. **Hosted PWA**: the same web app as static files on Cloudflare Pages, inference in the browser via WebGPU.

## Modes

The UX should be as easy as possible. Modes differ in how much control they show.

- **Live**: endless generation, played as it is made, tweakable while it plays (prompt, intensity, density,
  brightness). The whole session is recorded and can be saved.
- **Radio**: live mode driven by a station preset: one Driftscape ships with, one you saved, or one someone else made.
- **Producer**: a tiny, focused DAW. Segments on a timeline you can go back to, regenerate, extend, layer and export.

Live mode never stalls: the current segment loops seamlessly until the next one (a continuation of it, with the latest
tweaks) is ready, then a bar-aligned crossfade brings it in. Generation speed only decides how often a repeat happens.

## Genres

A genre is a plain TOML preset that informs every mode: a prompt template that wraps the user's words (a system prompt
for the model), the voice, and defaults for segment length, tempo, crossfade and sampling.

- **Realistic ambience**: rain, cave, campfire, medieval tavern. Needs a sound-effects-capable voice (Stable Audio
  Open), or a CC0 field-recording bed; MusicGen is music-only.
- **Synthetic ambience**: synth textures and melodic drones.
- **Ambient**: the genre itself.
- **Ambient techno**: ambient with a soft beat; fixed session tempo, segments stretched and cut on bars.
- **Thematic**: inspired by soundtracks of games, films and pop culture. The template translates a reference into
  style words (instrumentation, era, mood, production). Presets describe styles ("16-bit fantasy overworld", "80s synth
  sci-fi score") and never name or imitate a specific work.

## Decisions

- Voices are swappable. MusicGen small first (weights CC-BY-NC 4.0, so no paid tier with it); Stable Audio Open Small
  next (Stability Community License, free under $1M revenue).
- Stations and genres are files. Shipped ones live in the repo; community ones in a static catalogue. No accounts yet.
- Native inference via ONNX Runtime (`ort`), browser inference via onnxruntime-web; both load the same model files.

## Order of work

1. Core and native generation: loops, tempo, resampling (done); MusicGen via ort, model cache, `gen`, real-time factor.
2. Live and radio in the CLI: evolving loop, continuation, tweaks, recording, presets, time-stretch.
3. Realistic-ambience voice.
4. Web: hosted PWA and `driftscape serve`.
5. Producer mode: a rack of layers on a timeline; takes generated, regenerated, extended, trimmed, faded, exported
   (done in the web app; `driftscape render` mixes the same arrangement format).
6. Plugin.

Later, maybe: location-aware playback (soundscapes tied to places) and a platform for sharing, as separate consumers of
the engine.
