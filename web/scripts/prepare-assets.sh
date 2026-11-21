#!/usr/bin/env bash
# Build what the web app loads from static/:
#   wasm/driftscape.wasm   the Rust core (crates/wasm), with a size budget
#   presets.json           genres and stations, from the same TOML files the CLI uses
#   ort/                   ONNX Runtime's WebGPU loader, and its 28 MB wasm gzipped (Cloudflare Pages caps files at
#                          25 MiB; the browser unpacks it with DecompressionStream)
set -euo pipefail
cd "$(dirname "$0")/.."
root=..

BUDGET=${WASM_BUDGET_BYTES:-65536}
cargo build --release --target wasm32-unknown-unknown -p driftscape-wasm --manifest-path "$root/Cargo.toml"
in="$root/target/wasm32-unknown-unknown/release/driftscape_wasm.wasm"
mkdir -p static/wasm
if command -v wasm-opt >/dev/null; then wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int "$in" -o static/wasm/driftscape.wasm; else cp "$in" static/wasm/driftscape.wasm; fi
size=$(wc -c < static/wasm/driftscape.wasm | tr -d ' ')
echo "driftscape.wasm: $size bytes (budget $BUDGET)"
[ "$size" -le "$BUDGET" ] || { echo "over the size budget"; exit 1; }

cargo run --release -q -p driftscape-core --example presets --manifest-path "$root/Cargo.toml" > static/presets.json

# Three ONNX Runtime builds: WebGPU through its native (asyncify) implementation for Chrome, Edge and Safari; WebGPU
# through its JavaScript implementation (jsep), tried first in Firefox, where the native one crashes; and the CPU-only
# build as the fallback everywhere. Each wasm ships gzipped; the worker unpacks it.
ort=node_modules/onnxruntime-web/dist
mkdir -p static/ort
for build in ort-wasm-simd-threaded.asyncify ort-wasm-simd-threaded.jsep ort-wasm-simd-threaded; do
  cp "$ort/$build.mjs" static/ort/
  if [ ! -f "static/ort/$build.wasm.gz" ] || [ "$ort/$build.wasm" -nt "static/ort/$build.wasm.gz" ]; then
    gzip -9 -c "$ort/$build.wasm" > "static/ort/$build.wasm.gz"
  fi
  echo "$build: $(wc -c < "static/ort/$build.wasm.gz" | tr -d ' ') bytes gzipped"
done
