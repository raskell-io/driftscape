# Driftscape — Vision

A platform in the vein of Mixcloud and Soundcloud, but purpose-built for ambient music and soundscapes.

## Core Pillars

### 1. Ambient-first audio platform
Not a general music platform. Everything — UI, audio formats, social features — is shaped by the focus on ambient and background music. Music that is meant to be listened to in the background, is easily loopable, and can be extended to form soundscapes.

### 2. Minimal footprint
Highly compressed but still well-sounding audio. Ambient content is more forgiving of lossy compression than vocal or melodic music, so we can push bitrates lower (e.g. Opus at 32-64 kbps) while maintaining perceived quality. Loop metadata (sample-accurate start/end points) and app-layer crossfading handle gapless playback.

### 3. Composable soundscapes
Users can layer sound bites, environmental textures (beach waves, cicadas, rain, wind), and melodic loops into rich scenes. The platform should be friendly to sampling pre-existing "sound bites" — fragments from video games, famous melodies, environmental recordings — while keeping the overall focus on ambient/ambience genres.

### 4. Client-side AI generation
A web app that loads an AI audio model (MusicGen) into the browser via WebAssembly/WebGPU. All compute and GPU cycles happen on the client — no model serving on the backend. This keeps server costs flat and puts creative power directly in users' hands.

### 5. Geo-aware playback
People can curate soundscapes for specific locations. When a listener moves to a shop, a city, or a spot within a geographically bounded area, the music switches or shifts softly. Ranges from simple geofenced zones with assigned playlists to smooth spatial blending between zones (closer to a game audio engine).

### 6. Preference-driven experience
A wholesome, always-on background music experience tailored to individual taste. Some people want game-inspired music, others medieval/fantasy soundscapes, futuristic or cyberpunk atmospheres. The platform adapts to user profiles and preferences, ensuring an appropriate ambient backdrop at all times.

## Domain
`driftscape.live` or `driftscape.audio`

## First Milestone
**Client-side AI audio engine** — the riskiest and most novel piece. De-risk the hardest technical challenge first: getting MusicGen running in the browser via Transformers.js + WebGPU, with Web Audio API playback, seamless looping, and crossfade.

## Future Milestones
- User accounts and auth
- Audio upload and storage (S3-compatible)
- Soundscape composition (layering multiple loops and sound bites)
- Geo-aware playback with location-based soundscape switching
- Mobile PWA
- Social features (following, sharing, collaborative soundscapes)
