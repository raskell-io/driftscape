#!/usr/bin/env bash
# Build Driftscape as CLAP, VST3 and AUv2 into target/bundled/. `--install` copies them to ~/Library/Audio/Plug-Ins.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
cargo xtask bundle driftscape-plugin --release
clap="$root/target/bundled/Driftscape.clap"
build="$root/target/plugin-wrap"
cmake -S crates/plugin/wrap -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DDRIFTSCAPE_CLAP="$clap"
cmake --build "$build"
# clap-wrapper copies the AU's generated Info.plist (with its AudioComponents entry) before linking, and the Ninja
# bundle step can write CMake's plain one over it afterwards: put it back, then sign again (ad hoc).
cp "$build/driftscape_auv2-build-helper-output/auv2_Info.plist" "$build/Driftscape.component/Contents/Info.plist"
codesign -s - -f --deep "$build/Driftscape.component"
for b in Driftscape.vst3 Driftscape.component; do
  found=$(find "$build" -maxdepth 3 -name "$b" -type d | head -1)
  [ -n "$found" ] || { echo "no $b built"; exit 1; }
  rm -rf "$root/target/bundled/$b"
  cp -R "$found" "$root/target/bundled/$b"
done
ls -d "$root"/target/bundled/Driftscape.*
if [ "${1:-}" = "--install" ]; then
  dest="$HOME/Library/Audio/Plug-Ins"
  mkdir -p "$dest/CLAP" "$dest/VST3" "$dest/Components"
  rm -rf "$dest/CLAP/Driftscape.clap" "$dest/VST3/Driftscape.vst3" "$dest/Components/Driftscape.component"
  cp -R target/bundled/Driftscape.clap "$dest/CLAP/"
  cp -R target/bundled/Driftscape.vst3 "$dest/VST3/"
  cp -R target/bundled/Driftscape.component "$dest/Components/"
  echo "installed to $dest"
fi
