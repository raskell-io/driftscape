//! `driftscape serve`: the web app, on this machine, with the native engine behind it.
//!
//! Serves the built web app (web/build, embedded in release builds) and two endpoints the app looks for:
//! - `GET /api/health` tells the app a native engine is here, so it uses it instead of WebGPU in the browser;
//! - `POST /api/generate` takes the same request the in-browser voice takes (web/src/lib/voice/voice.js) and returns a
//!   JSON line, a newline, then the samples as little-endian f32.
//!
//! The app has one generation in flight at a time, so a new request cancels any older one still running: that is how
//! a tweak cancels the stale segment without the server having to notice a dropped connection.

use crate::generate::models_dir;
use anyhow::{Context, Result, anyhow};
use driftscape_core::{Audio, looping::rejoin_point, tempo};
use driftscape_infer::{
    ControlFlow, Request, Tokens, Voice, is_cancelled,
    musicgen::{Backend, MusicGen},
};
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use tiny_http::{Header, Method, Response, Server};

#[derive(RustEmbed)]
#[folder = "../../web/build"]
struct WebApp;

const SAMPLE_RATE: u32 = 32_000;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateRequest {
    prompt: String,
    seconds: f32,
    guidance: f32,
    temperature: f32,
    top_k: usize,
    seed: f64,
    continue_from: Option<Vec<Vec<i64>>>,
    analysis: Analysis,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Analysis {
    crossfade_seconds: f32,
    bpm: Option<f32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerateResult {
    sample_rate: u32,
    codebooks: Vec<Vec<i64>>,
    overlap_seconds: f32,
    rejoin: usize,
    bpm: f64,
    seconds: f32,
}

pub fn run(port: u16, backend: Backend, open: bool) -> Result<()> {
    if WebApp::get("index.html").is_none() {
        anyhow::bail!("the web app isn't built into this binary; run `npm run build` in web/ and rebuild driftscape");
    }
    eprintln!("loading the engine…");
    let env = ort::environment::init().with_name("driftscape").with_telemetry(false).build()?;
    let voice = Arc::new(Mutex::new(MusicGen::load(&env, &models_dir()?.join("musicgen-small"), backend)?));
    let latest = Arc::new(AtomicU64::new(0));

    let server = Server::http(("127.0.0.1", port)).map_err(|e| anyhow!("listening on port {port}: {e}"))?;
    let url = format!("http://localhost:{port}/");
    eprintln!("Driftscape is at {url} (Ctrl-C to stop)");
    if open {
        let _ =
            std::process::Command::new(if cfg!(target_os = "macos") { "open" } else { "xdg-open" }).arg(&url).spawn();
    }

    for request in server.incoming_requests() {
        let (voice, latest) = (voice.clone(), latest.clone());
        std::thread::spawn(move || {
            if let Err(e) = handle(request, &voice, &latest) {
                eprintln!("request failed: {e:#}");
            }
        });
    }
    Ok(())
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid header")
}

fn handle(mut request: tiny_http::Request, voice: &Mutex<MusicGen>, latest: &AtomicU64) -> Result<()> {
    let path = request.url().split('?').next().unwrap_or("/").to_owned();
    match (request.method(), path.as_str()) {
        (Method::Get, "/api/health") => {
            let body = serde_json::json!({ "engine": "driftscape", "version": env!("CARGO_PKG_VERSION") }).to_string();
            request.respond(Response::from_string(body).with_header(header("content-type", "application/json")))?;
        }
        (Method::Post, "/api/generate") => {
            let id = latest.fetch_add(1, Ordering::AcqRel) + 1;
            let mut body = String::new();
            request.as_reader().read_to_string(&mut body)?;
            let req: GenerateRequest = serde_json::from_str(&body).context("reading the request")?;
            match generate(&req, voice, || latest.load(Ordering::Acquire) != id) {
                Ok((meta, samples)) => {
                    let mut out = serde_json::to_vec(&meta)?;
                    out.push(b'\n');
                    out.reserve(samples.len() * 4);
                    for s in samples {
                        out.extend_from_slice(&s.to_le_bytes());
                    }
                    request.respond(
                        Response::from_data(out).with_header(header("content-type", "application/octet-stream")),
                    )?;
                }
                Err(e) if is_cancelled(&e) => {
                    request.respond(Response::from_string("cancelled").with_status_code(499))?
                }
                Err(e) => request.respond(Response::from_string(format!("{e:#}")).with_status_code(500))?,
            }
        }
        (Method::Get, _) => {
            let name = path.trim_start_matches('/');
            let name = if name.is_empty() { "index.html" } else { name };
            let file =
                WebApp::get(name).or_else(|| WebApp::get(&format!("{name}.html"))).or_else(|| WebApp::get("404.html"));
            match file {
                Some(f) => {
                    let mime = f.metadata.mimetype().to_owned();
                    request.respond(
                        Response::from_data(f.data.into_owned())
                            .with_header(header("content-type", &mime))
                            .with_header(header("cross-origin-opener-policy", "same-origin"))
                            .with_header(header("cross-origin-embedder-policy", "require-corp"))
                            .with_header(header("cache-control", "no-cache")),
                    )?;
                }
                None => request.respond(Response::from_string("not found").with_status_code(404))?,
            }
        }
        _ => request.respond(Response::from_string("method not allowed").with_status_code(405))?,
    }
    Ok(())
}

/// Run one request on the native voice; `stale` turns true when a newer request arrives.
fn generate(
    req: &GenerateRequest,
    voice: &Mutex<MusicGen>,
    stale: impl Fn() -> bool,
) -> Result<(GenerateResult, Vec<f32>)> {
    let started = std::time::Instant::now();
    let request = Request {
        guidance: req.guidance,
        temperature: req.temperature,
        top_k: req.top_k,
        continue_from: req.continue_from.clone().map(|codebooks| Tokens { codebooks, frame_rate: 50.0 }),
        ..Request::new(req.prompt.clone(), req.seconds, req.seed as u64)
    };
    // Wait for the voice; a request that went stale while waiting gives up without generating.
    let mut voice = voice.lock().map_err(|_| anyhow!("the engine crashed earlier; restart driftscape serve"))?;
    if stale() {
        return Err(driftscape_infer::Cancelled.into());
    }
    let clip =
        voice.generate(&request, &mut |_| if stale() { ControlFlow::Break(()) } else { ControlFlow::Continue(()) })?;
    drop(voice);

    let overlap = (clip.overlap_seconds * SAMPLE_RATE as f32) as usize;
    let (rejoin, bpm) = analyse(&clip.audio, overlap, &req.analysis);
    let meta = GenerateResult {
        sample_rate: SAMPLE_RATE,
        codebooks: clip.tokens.codebooks,
        overlap_seconds: clip.overlap_seconds,
        rejoin,
        bpm,
        seconds: started.elapsed().as_secs_f32(),
    };
    Ok((meta, clip.audio.channels.into_iter().next().unwrap_or_default()))
}

/// The same analysis the in-browser voice does (web/src/lib/voice/musicgen.worker.js), natively.
fn analyse(audio: &Audio, overlap: usize, analysis: &Analysis) -> (usize, f64) {
    let samples = &audio.channels[0];
    let rate = SAMPLE_RATE as usize;
    let crossfade = ((analysis.crossfade_seconds * SAMPLE_RATE as f32) as usize).min(samples.len());
    let window = crossfade.min(rate / 2).max(1);
    let from = samples.len() - crossfade;
    let latest = from.saturating_sub(4 * rate);
    let bar = analysis.bpm.map(|b| (4.0 * 60.0 / f64::from(b) * rate as f64).round() as usize);
    let rejoin = rejoin_point(samples, from, window, overlap.min(latest), latest, bar)
        .or_else(|| rejoin_point(samples, from, window, 0, latest, None))
        .map_or(overlap, |(s, _)| s);
    let bpm = match analysis.bpm {
        Some(_) => tempo::estimate(samples, SAMPLE_RATE, tempo::Range::default()).map_or(0.0, |t| t.bpm),
        None => 0.0,
    };
    (rejoin, bpm)
}
