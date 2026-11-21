//! `driftscape serve` embeds the built web app from web/build. Make sure the folder exists so the CLI also builds
//! without Node; serve then explains how to build the app.
fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/build");
    if !dir.exists() {
        let _ = std::fs::create_dir_all(&dir);
    }
    println!("cargo:rerun-if-changed=../../web/build");
}
