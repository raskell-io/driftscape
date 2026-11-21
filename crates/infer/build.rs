//! ONNX Runtime's prebuilt CoreML code calls `__isPlatformVersionAtLeast` (an `@available` check), which lives in
//! clang's builtins library. Release builds with LTO don't pull that library in on their own, so link it explicitly.
use std::process::Command;

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let dir = Command::new("clang").arg("--print-runtime-dir").output().ok().filter(|o| o.status.success());
    match dir {
        Some(o) => {
            let dir = String::from_utf8_lossy(&o.stdout).trim().to_owned();
            println!("cargo:rustc-link-search=native={dir}");
            println!("cargo:rustc-link-lib=static=clang_rt.osx");
        }
        None => println!("cargo:warning=clang not found; release builds may fail to link ONNX Runtime"),
    }
}
