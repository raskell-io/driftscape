//! Driftscape as a DAW instrument: live mode inside the host. The session engine ([`driftscape_engine`]) generates
//! on its own threads; the host's audio callback only pulls mixed samples from it.
//!
//! - It plays while the host's transport runs (or always, with "Free run"), and pauses with it: the engine's clock
//!   is the samples pulled, so nothing is skipped.
//! - For genres with a tempo it asks the model for the host's tempo, which MusicGen follows closely.
//! - Genre, focus and wander are host parameters (automatable). A change reaches the engine through nih-plug's
//!   background task executor, because changing it takes a lock and `process` must never wait.
//! - The mood is free text in the editor, saved with the project.
//!
//! Exported as CLAP. VST3 and AU come from wrapping the CLAP with clap-wrapper (`plugin/wrap/`).

use driftscape_core::preset::{Genre, builtin_genres};
use driftscape_engine::{AudioSide, Config, Event, Params as EngineParams, Session, VoiceBackend, listener_or_default};
use nih_plug::prelude::*;
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};
use std::sync::{Arc, Mutex, RwLock};

mod style;

pub struct Driftscape {
    params: Arc<DriftscapeParams>,
    genres: Arc<Vec<Genre>>,
    /// The session, shared with the editor and the background task. Never locked in `process`.
    engine: Arc<Mutex<Option<Session>>>,
    /// The real-time half of the session, owned by the audio thread.
    audio: Option<AudioSide>,
    scratch: Vec<f32>,
    /// What the engine was last told (genre index, focus, wander, tempo), to notice changes in `process`.
    sent: Option<Steer>,
    status: Arc<Mutex<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Steer {
    genre: usize,
    focus: f32,
    wander: f32,
    /// The host's tempo, for genres with a beat (rounded to a tenth of a BPM).
    bpm: Option<f32>,
}

/// Work done off the audio thread.
pub enum Task {
    Steer(Steer),
}

#[derive(Params)]
pub struct DriftscapeParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
    /// The listener's words. Empty means the genre's own prompt.
    #[persist = "mood"]
    mood: Arc<RwLock<String>>,

    #[id = "genre"]
    pub genre: IntParam,
    /// Added to the genre's guidance, in half steps: how closely to follow the mood.
    #[id = "focus"]
    pub focus: FloatParam,
    /// Added to the genre's temperature, in tenths: how much the music wanders.
    #[id = "wander"]
    pub wander: FloatParam,
    #[id = "level"]
    pub level: FloatParam,
    /// Ask for the host's tempo (genres with a beat).
    #[id = "tempo"]
    pub follow_tempo: BoolParam,
    /// Play even while the host's transport is stopped.
    #[id = "free"]
    pub free_run: BoolParam,
}

impl DriftscapeParams {
    fn new(genres: &[Genre]) -> Self {
        let names: Vec<String> = genres.iter().map(|g| g.name.clone()).collect();
        let count = names.len() as i32;
        Self {
            editor_state: EguiState::from_size(560, 380),
            mood: Arc::new(RwLock::new(String::new())),
            genre: IntParam::new("Genre", 0, IntRange::Linear { min: 0, max: count - 1 })
                .with_value_to_string(Arc::new(move |i| names.get(i as usize).cloned().unwrap_or_default()))
                .non_automatable(),
            focus: FloatParam::new("Focus", 0.0, FloatRange::Linear { min: -2.0, max: 2.0 }).with_step_size(0.5),
            wander: FloatParam::new("Wander", 0.0, FloatRange::Linear { min: -2.0, max: 2.0 }).with_step_size(0.5),
            level: FloatParam::new(
                "Level",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-48.0),
                    max: util::db_to_gain(6.0),
                    factor: FloatRange::gain_skew_factor(-48.0, 6.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(1))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            follow_tempo: BoolParam::new("Follow tempo", true),
            free_run: BoolParam::new("Free run", false),
        }
    }
}

impl Default for Driftscape {
    fn default() -> Self {
        let genres = builtin_genres();
        Self {
            params: Arc::new(DriftscapeParams::new(&genres)),
            genres: Arc::new(genres),
            engine: Arc::new(Mutex::new(None)),
            audio: None,
            scratch: Vec::new(),
            sent: None,
            status: Arc::new(Mutex::new("Starting".into())),
        }
    }
}

impl Driftscape {
    fn steer_now(&self, transport_tempo: Option<f64>) -> Steer {
        let genre = (self.params.genre.value().max(0) as usize).min(self.genres.len() - 1);
        let bpm = match (self.genres[genre].bpm, self.params.follow_tempo.value(), transport_tempo) {
            (Some(_), true, Some(t)) if (40.0..=220.0).contains(&t) => Some((t * 10.0).round() as f32 / 10.0),
            _ => None,
        };
        Steer { genre, focus: self.params.focus.value(), wander: self.params.wander.value(), bpm }
    }
}

/// The engine's parameters for a steer and a mood.
fn engine_params(genres: &[Genre], steer: Steer, mood: &str) -> EngineParams {
    let mut genre = genres[steer.genre].clone();
    if let (Some(_), Some(bpm)) = (genre.bpm, steer.bpm) {
        genre.bpm = Some(bpm);
    }
    EngineParams { genre, prompt: mood.to_owned(), focus: steer.focus * 0.5, variety: steer.wander * 0.1 }
}

impl Plugin for Driftscape {
    const NAME: &'static str = "Driftscape";
    const VENDOR: &'static str = "raskell.io";
    const URL: &'static str = "https://driftscape.raskell.io";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: None,
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];
    // Instruments in Logic need a MIDI input; notes are ignored for now.
    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;

    type SysExMessage = ();
    type BackgroundTask = Task;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let (engine, genres, params) = (self.engine.clone(), self.genres.clone(), self.params.clone());
        Box::new(move |task| match task {
            Task::Steer(steer) => {
                let mood = params.mood.read().map(|m| m.clone()).unwrap_or_default();
                if let Some(session) = engine.lock().unwrap().as_ref() {
                    let next = engine_params(&genres, steer, &mood);
                    session.update(|p| *p = next);
                }
            }
        })
    }

    fn initialize(
        &mut self,
        _layout: &AudioIOLayout,
        config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.scratch = vec![0.0; config.max_buffer_size as usize];
        let steer = self.steer_now(None);
        let mood = self.params.mood.read().map(|m| m.clone()).unwrap_or_default();
        // A new sample rate means a new session; the old one stops first (at most one model step).
        let mut engine = self.engine.lock().unwrap();
        *engine = None;
        self.audio = None;
        match Session::start(Config {
            params: engine_params(&self.genres, steer, &mood),
            station: None,
            backend: VoiceBackend::Gpu,
            rate: config.sample_rate.round() as u32,
        }) {
            Ok((session, audio)) => {
                *engine = Some(session);
                self.audio = Some(audio);
                self.sent = Some(steer);
                *self.status.lock().unwrap() = "Loading the model".into();
            }
            Err(e) => {
                nih_log!("Driftscape couldn't start: {e:#}");
                *self.status.lock().unwrap() = format!("{e:#}");
            }
        }
        drop(engine);
        let _ = context;
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let transport = context.transport();
        let running = transport.playing || self.params.free_run.value();
        let steer = self.steer_now(transport.tempo);
        if self.audio.is_some() && self.sent != Some(steer) {
            self.sent = Some(steer);
            context.execute_background(Task::Steer(steer));
        }

        let n = buffer.samples();
        let Some(audio) = self.audio.as_mut().filter(|_| running && n <= self.scratch.len()) else {
            for channel in buffer.as_slice() {
                channel.fill(0.0);
            }
            return ProcessStatus::Normal;
        };
        let mono = &mut self.scratch[..n];
        audio.process(mono);
        for (i, mut frame) in buffer.iter_samples().enumerate() {
            let gain = self.params.level.smoothed.next();
            for s in frame.iter_mut() {
                *s = mono[i] * gain;
            }
        }
        ProcessStatus::KeepAlive
    }

    fn editor(&mut self, _executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let (params, engine, genres, status) =
            (self.params.clone(), self.engine.clone(), self.genres.clone(), self.status.clone());
        let draft = params.mood.read().map(|m| m.clone()).unwrap_or_default();
        create_egui_editor(
            self.params.editor_state.clone(),
            draft,
            |ctx, _| style::apply(ctx),
            move |ctx, setter, draft| {
                // The engine's news, shown in the status line.
                if let Some(session) = engine.lock().unwrap().as_ref() {
                    while let Some(event) = session.try_event() {
                        let text = match event {
                            Event::Loaded => "Making the first part".to_owned(),
                            Event::Playing { label } => label,
                            Event::Repeating => "Repeating while the next part is made".to_owned(),
                            Event::Paused => "Paused with the transport".to_owned(),
                            Event::Resumed => "Playing".to_owned(),
                            Event::Failed(e) => format!("Stopped: {e}"),
                            Event::Trace(_) => continue,
                        };
                        *status.lock().unwrap() = text;
                    }
                }
                ctx.request_repaint_after(std::time::Duration::from_millis(250));

                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("DRIFTSCAPE").strong().size(15.0).color(style::PEACH));
                    ui.add_space(10.0);

                    let genre = &genres[(params.genre.value().max(0) as usize).min(genres.len() - 1)];
                    egui::Grid::new("controls").num_columns(2).spacing([18.0, 10.0]).show(ui, |ui| {
                        ui.label("Genre");
                        egui::ComboBox::from_id_salt("genre").selected_text(&genre.name).width(240.0).show_ui(
                            ui,
                            |ui| {
                                for (i, g) in genres.iter().enumerate() {
                                    if ui.selectable_label(params.genre.value() == i as i32, &g.name).clicked() {
                                        setter.begin_set_parameter(&params.genre);
                                        setter.set_parameter(&params.genre, i as i32);
                                        setter.end_set_parameter(&params.genre);
                                    }
                                }
                            },
                        );
                        ui.end_row();

                        ui.label("Mood");
                        let field = ui.add(
                            egui::TextEdit::singleline(draft).hint_text(&genre.default_prompt).desired_width(300.0),
                        );
                        if field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            if let Ok(mut m) = params.mood.write() {
                                *m = draft.clone();
                            }
                            if let Some(session) = engine.lock().unwrap().as_ref() {
                                let words = draft.clone();
                                session.update(|p| p.prompt = words);
                            }
                        }
                        ui.end_row();

                        ui.label("Focus");
                        ui.add(widgets::ParamSlider::for_param(&params.focus, setter).with_width(240.0));
                        ui.end_row();
                        ui.label("Wander");
                        ui.add(widgets::ParamSlider::for_param(&params.wander, setter).with_width(240.0));
                        ui.end_row();
                        ui.label("Level");
                        ui.add(widgets::ParamSlider::for_param(&params.level, setter).with_width(240.0));
                        ui.end_row();
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        toggle(ui, setter, &params.follow_tempo, "Follow the host's tempo");
                        toggle(ui, setter, &params.free_run, "Play while stopped");
                    });
                    ui.add_space(12.0);
                    ui.separator();
                    let line = status.lock().unwrap().clone();
                    ui.label(egui::RichText::new(line).monospace().size(11.0).color(style::APRICOT));
                    ui.label(
                        egui::RichText::new(format!(
                            "{} · {}",
                            genre.name,
                            listener_or_default(&params.mood.read().map(|m| m.clone()).unwrap_or_default(), genre)
                        ))
                        .size(11.0)
                        .color(style::MUTED),
                    );
                });
            },
        )
    }
}

fn toggle(ui: &mut egui::Ui, setter: &ParamSetter, param: &BoolParam, label: &str) {
    let mut on = param.value();
    if ui.checkbox(&mut on, label).changed() {
        setter.begin_set_parameter(param);
        setter.set_parameter(param, on);
        setter.end_set_parameter(param);
    }
}

impl ClapPlugin for Driftscape {
    const CLAP_ID: &'static str = "io.raskell.driftscape";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Endless ambient music, generated on this computer");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] =
        &[ClapFeature::Instrument, ClapFeature::Synthesizer, ClapFeature::Stereo];
}

nih_export_clap!(Driftscape);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_tempo_only_applies_to_genres_with_a_beat() {
        let plugin = Driftscape::default();
        let techno = plugin.genres.iter().position(|g| g.bpm.is_some()).expect("a genre with a beat");
        let ambient = plugin.genres.iter().position(|g| g.bpm.is_none()).expect("a genre without one");
        let p = engine_params(&plugin.genres, Steer { genre: techno, focus: 1.0, wander: -1.0, bpm: Some(124.0) }, "");
        assert_eq!(p.genre.bpm, Some(124.0));
        assert_eq!((p.focus, p.variety), (0.5, -0.1));
        let p =
            engine_params(&plugin.genres, Steer { genre: ambient, focus: 0.0, wander: 0.0, bpm: Some(124.0) }, "rain");
        assert_eq!(p.genre.bpm, None);
        assert_eq!(p.prompt, "rain");
    }
}
