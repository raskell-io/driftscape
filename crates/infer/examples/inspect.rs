//! Print the inputs and outputs of every ONNX file in a model directory.
fn main() -> ort::Result<()> {
    let env = ort::environment::init().with_telemetry(false).build()?;
    let dir = std::env::args().nth(1).expect("model directory");
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "onnx") {
            let s = ort::session::Session::builder(&env)?.commit_from_file(&path)?;
            println!("== {}", path.display());
            for i in s.inputs().iter().filter(|i| !i.name().starts_with("past")) {
                println!("  in  {} {}", i.name(), i.dtype());
            }
            if s.inputs().len() > 8 {
                println!("  ... {} inputs", s.inputs().len());
            }
            for o in s.outputs().iter().take(4) {
                println!("  out {} {}", o.name(), o.dtype());
            }
            if s.outputs().len() > 4 {
                println!("  ... {} outputs", s.outputs().len());
            }
        }
    }
    Ok(())
}
