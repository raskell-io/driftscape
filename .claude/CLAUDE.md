# Driftscape

A local AI music engine for ambient soundscapes. Inference always runs on the user's own machine. One Rust core,
several ways in: a CLI (primary), a DAW plugin, a local web GUI (`driftscape serve`) and a hosted static PWA.
The plan, with diagrams, modes and genres: `.claude/VISION.md`.

The Phoenix LiveView prototype lives on the `platform` branch.

## Layout

```
crates/core/   pure audio work: WAV, resampling, tempo, WSOLA time-stretch, seamless loops, tempo-fitted loops
               (fit_loop) and rejoin points, the real-time mixer and
               leveler, genre/station presets, producer arrangements and their mixdown (arrange.rs).
               No I/O, no model code, no threads.
crates/infer/  voices. MusicGen: text encoder + EnCodec on ONNX Runtime (q4 files from Xenova/musicgen-small),
               decoder on candle/Metal (gpu.rs, facebook/musicgen-small safetensors) or ONNX on CPU (onnx.rs)
crates/cli/    the `driftscape` binary: live, radio, gen, loop, render, play, info, serve. live.rs runs live and radio mode
               (generator thread, control loop, audio callback, recorder thread).
crates/wasm/   C ABI over core for the browser (rejoin, tempo, arr_* for the producer mixdown); no wasm-bindgen, no
               serde_json (the 64 KB budget): the arrangement goes in as arr_track/arr_clip calls
presets/       genres/*.toml and stations/*.toml, compiled into core (preset.rs lists them)
web/           SvelteKit app (adapter-static, Svelte 5, #lib/* imports). lib/engine.js ports live.rs to Web Audio;
               lib/voice/ has the in-browser MusicGen worker (onnxruntime-web, WebGPU), the T5 tokenizer, and
               NativeVoice for `driftscape serve` (crates/cli/src/serve.rs embeds web/build)
```

Models live in `~/Library/Application Support/Driftscape/models/musicgen-small/` (`onnx/`, `tokenizer.json`,
`safetensors/model.safetensors`).

Planned: `crates/plugin` (nih-plug CLAP, wrapped to VST3/AU with clap-wrapper), `web/` (SvelteKit, adapter-static).

## Live mode

- Each segment continues from the previous one's last `overlap_seconds` of tokens, so its first frames reproduce
  the previous tail (measured: 0.99 correlation, zero lag; `--example continuation`). The crossfade sits in that overlap.
- If the next segment is late, the control loop crossfades back to the current segment's `rejoin` point (bar-aligned
  for genres with `bpm`). The next segment still continues from the buffer's end, which comes round again.
- The audio callback only applies commands and mixes; buffers are freed on the control thread.
- A tweak bumps a version counter: the waiting segment is dropped, the one being generated is cancelled (via the
  progress callback), and the next is made from the *playing* segment's tokens. It is short (6 s), marked urgent, and
  crossfades in (3 s) as soon as it's ready. Measured delay: 5–10 s.
- Segments stay full length otherwise: each replays the overlap first, so short segments lose throughput. The overlap
  is fed as one batch (`Decoder::prefill`; GPU: 150 positions in ~160 ms instead of ~1.35 s; `--example prefill`
  checks it matches stepping).
- The control loop warns if the audio clock stops (device change). `:quit` and Ctrl-C never wait on a lock; a second
  Ctrl-C exits at once.
- Open: one session (2026-11-21) went silent after a urgent transition and ignored `:quit` for ~55 min. Not
  reproduced since; the hardening above addresses the plausible causes. `DRIFTSCAPE_TRACE=1` prints a clock heartbeat.
- Tempo: MusicGen follows a BPM in the prompt closely (118 asked → 118.0 measured, every segment of a 2.5 min
  night-drive session, and 118.01–118.03 in `gen`). Live mode therefore does not stretch segments (crossfading two
  stretched copies would flam); `fit_loop`/`stretch` are for fixed outside tempos: `gen --bpm --bars`, `loop
  --to-bpm`, and later the plugin following the host.
- Adding a preset: drop a TOML file in `presets/` and list it in `crates/core/src/preset.rs`; the tests check it.

## Web notes

- `web/scripts/prepare-assets.sh` builds the core wasm, exports presets (`driftscape presets --json`), and ships
  onnxruntime-web's 28 MB WebGPU runtime gzipped (Cloudflare Pages caps files at 25 MiB). The worker sniffs the gzip
  magic because some servers send `.gz` with `Content-Encoding: gzip` and the browser unpacks it first.
- Vite resolves `onnxruntime-web` with the `onnxruntime-web-use-extern-wasm` condition; the default build makes Vite
  copy the 28 MB wasm into the output.
- onnxruntime-web refuses to create WebGPU sessions concurrently: create them one by one.
- The tokenizer is checked against the Rust `tokenizers` crate (`npm test`; fixtures from `--example tokenize`).
- `vite preview` does not pick up a rebuild; restart it. Unregister the service worker when testing a new build.
- Hosting: Cloudflare Pages, Git-connected to raskell-io/driftscape: root directory `web`, build command
  `npm run build`, output `build`, NODE_VERSION 24. The image has no Rust: prepare-assets.sh installs rustup, which reads rust-toolchain.toml. Presets are exported by
  `cargo run -p driftscape-core --example presets` so the Pages build never compiles the CLI (ONNX Runtime, ALSA).
  Domain: driftscape.raskell.io. `mise run deploy` uploads a local build by hand.
- App icons are rendered from static/icon.svg with headless Chrome (ImageMagick renders this SVG extremely slowly).
- Persistence (web/src/lib/store.js, IndexedDB): settings; a snapshot of the playing segment (audio + tokens) so
  "Continue where you left off" plays at once and generates on from its tokens; recordings written in 2 s chunks, so a
  reload or crash keeps them (listed as interrupted). The model is in Cache Storage (`driftscape-models-v1`); the page
  calls `navigator.storage.persist()` after loading.
- Service worker: a new deploy waits; the page shows "A new version is ready · Reload" and only then sends
  `skip-waiting`. Never reload on `controllerchange` unless the listener asked (a first install also fires it).
- Browser backends (web/src/lib/voice): `webgpu` = onnxruntime-web's native WebGPU (asyncify build; Chrome, Edge,
  Safari), `webgpu-js` = its JavaScript WebGPU (jsep build; tried first in Firefox, where the native one crashes with a
  wasm memory fault), `cpu` = wasm (fallback everywhere, about 0.1x real time; threads need COOP+COEP). `?engine=gpu|jsep|cpu`
  forces one. A crash or failed self-check falls back to the CPU.
- Self-check at load (it doubles as the warm-up): decode reference tokens and compare level and zero-crossing rate with
  the native decode (selfcheck.reference.json, from `--example greedy -- <dir> cpu --json`); if the audio decoder is
  wrong on the GPU, it moves to the CPU. Then greedy decoding must open with the reference tokens. Found: the jsep
  WebGPU build decodes EnCodec wrongly (same tokens, 3x the level, not music) while its decoder is right.
- fp16 (`?precision=f16`, Xenova's q4f16 files, two decoders) fails the self-check: wrong first token, degenerate
  codebooks, integer overflow on continuation. Opt-in only; the default chain never downloads it.
- In the browser generation runs ~0.45x real time (Chrome, M4), so about half of what plays is repeats. Repeats go back
  >= 8 s where the segment allows, rotate through up to three rejoin points, and crossfade up to 4 s (never longer than
  the audio before the rejoin point). Urgent short segments are only used when the measured speed keeps up (>= 0.9x);
  otherwise segments stay full length (a short one just loops). `localStorage["driftscape-trace"] = 1` logs repeats.
- Test with sound off: set the Level knob to 0 (settings in IndexedDB) and read the trace; never play audio unannounced.
- Never trust "it runs" for audio: check output against the native reference before calling a backend working.
- UI (chosen 2026-11-21; see the concept page https://claude.ai/artifact/5FKYXGJzmyuxeDDCmRGKmr): instruments, calm
  rather than skeuomorphic. Live = Terrain (web/src/lib/ui/Terrain.svelte, logic in terrain.js): genres are hills on a
  contour map, the dot's position gives a genre plus a blended second one (Genre.flavor words). Dragging steers urgently;
  the Drift knob's autopilot steers gently (engine.tweak(..., {urgent: false}) only shapes the next part). Radio =
  Receiver.svelte: a tuning dial over a band; resting on a station for 0.7 s tunes in. One dark theme.
- Inline `style` attributes are blocked by the CSP: set CSS variables from script (`el.style.setProperty`).
- Producer = Rack.svelte, logic in lib/producer/project.js (pure, tested), mixdown in Rust via lib/producer/mixdown.js
  (the main thread has its own instance of the core wasm). Takes are in IndexedDB ("takes", schema v2), the project in
  kv "project". Preview mixes at 32 kHz, export at 48 kHz 24-bit stereo. Extend continues a take's tokens: the new take
  starts `overlapSeconds` before the old one ends and they crossfade linearly (measured 0.99 correlation there).
  `Curve::Linear` for aligned material, equal power for unrelated. Grid items need `minmax(0, 1fr)` columns, or the
  wide timeline stretches the instrument.
- The app picks NativeVoice when `/api/health` answers `{"engine":"driftscape"}`, else WorkerVoice.

## Rules

- Logic goes in `crates/core`. It stays target-independent so it runs natively and as wasm: no `std::fs`, `std::net`,
  threads, `wasm_bindgen` or `web_sys` there.
- Audio is planar `f32` (`driftscape_core::Audio`). MusicGen outputs 32 kHz mono.
- Every DSP change gets a test with a synthetic signal that shows the property (no click at the seam, tempo within
  0.1 BPM, no aliasing).
- Never block an audio callback: resample and prepare buffers before the stream starts; the callback only copies.

## Inference notes

- `cargo run --release -p driftscape-infer --features metal --example parity -- <model dir>` compares the GPU decoder's
  logits with the ONNX one. Step 0 must correlate at 1.00; later steps ~0.93–0.98 (q4 noise). Run it after any change
  to `gpu.rs`.
- `--example stepcost <model dir> [position]` splits a GPU step into CPU queueing and GPU time at a sequence
  position. Real time needs < 20 ms per step; currently ~9 ms at 20, ~10.5 ms at 400, ~12 ms at 1000 (M4, q8_0).
  Check a late position too: per-step costs that grow with the sequence hide in early-position measurements.
- Env knobs: `DRIFTSCAPE_TRACE=1` (stage timings), `DRIFTSCAPE_QUANT=q8_0|q4k|none` (GPU weight quantization, default q8_0; q4k loses accuracy),
  `DRIFTSCAPE_DTYPE=f16|bf16` (GPU precision when QUANT=none),
  `DRIFTSCAPE_THREADS=n` (ONNX CPU threads).
- `crates/infer/build.rs` links clang's runtime: ONNX Runtime's CoreML code needs `__isPlatformVersionAtLeast`.

## Commands

```bash
mise run test                     # cargo test, clippy -D warnings, fmt --check
cargo run -p driftscape -- --help
```
