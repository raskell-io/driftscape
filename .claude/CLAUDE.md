# Driftscape

Ambient soundscape platform with a client-side AI audio engine powered by MusicGen.

## Stack

- **Backend**: Elixir / Phoenix 1.8
- **Frontend**: Phoenix LiveView 1.1 + JS hooks (Web Audio API, Transformers.js)
- **Database**: SQLite via ecto_sqlite3
- **AI Model**: MusicGen small (Xenova/musicgen-small, q4 quantized) via Transformers.js + ONNX Runtime Web (WebGPU backend)
- **Audio**: Web Audio API (AudioBufferSourceNode with looping and crossfade)
- **Asset pipeline**: esbuild (Phoenix default)

## Dev Setup

```bash
mix setup        # deps, DB create/migrate, asset build
mix phx.server   # start dev server at localhost:4000
```

## Key Directories

```
lib/driftscape/           # Business logic, Repo, schemas
lib/driftscape_web/       # Phoenix web layer
  live/engine_live.ex     # AI audio engine LiveView (/engine)
  plugs/                  # Custom plugs (COOP/COEP)
  components/             # Phoenix components
assets/
  js/hooks/               # LiveView JS hooks
    audio_engine.js       # Transformers.js + Web Audio integration
    index.js              # Hook aggregator
  package.json            # npm deps (@huggingface/transformers)
priv/static/wasm/         # WASM files served statically
```

## Architecture Notes

- **COOP/COEP headers** are set globally via `CrossOriginIsolation` plug — required for SharedArrayBuffer (AudioWorklet ↔ Worker comms)
- **AI inference runs entirely client-side** — no model serving on the backend
- **MusicGen model** is lazy-loaded on first generate, cached by the browser
- **Audio playback** uses seamless looping with GainNode-based crossfade between clips

## Testing

```bash
mix test                # run all tests
mix test --cover        # with coverage
mix precommit           # compile (warnings-as-errors) + format + test
```

## Conventions

- Binary IDs (UUIDs) for all Ecto schemas
- UTC timestamps on all schemas
- Phoenix generators follow these defaults (set in mix.exs)
