//! Print the Rust tokenizer's ids for test prompts as JSON: the reference the web tokenizer is tested against.
fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("model directory"));
    let t = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompts = [
        "warm ambient pads",
        "dusk over a quiet harbour. Ambient music, slow evolving pads, soft warm textures, gentle harmony, spacious reverb, no drums",
        "Ambient techno at 118 BPM, soft deep kick drum on every beat",
        "16-bit fantasy role-playing game overworld, gentle flute and harp, chiptune textures",
        "  rain   on a tin roof,\tsoft guitar  ",
        "café at 4am — neon, 80s synthwave; ÉLAN!",
        "medieval tavern, crackling fireplace (quiet murmur) & rain",
    ];
    let cases: Vec<String> = prompts
        .iter()
        .map(|p| {
            let ids = t.encode(*p, true).unwrap().get_ids().to_vec();
            format!("  {{ \"prompt\": {:?}, \"ids\": {:?} }}", p, ids)
        })
        .collect();
    println!("[\n{}\n]", cases.join(",\n"));
    Ok(())
}
