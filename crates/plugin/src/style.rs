//! The editor's look: Driftscape's dusk palette (as the web app), calm and dark.

use nih_plug_egui::egui::{self, Color32, CornerRadius, Stroke};

pub const NIGHT: Color32 = Color32::from_rgb(0x18, 0x14, 0x2c);
pub const PANEL: Color32 = Color32::from_rgb(0x22, 0x1d, 0x3b);
pub const LINE: Color32 = Color32::from_rgb(0x2d, 0x26, 0x50);
pub const INK: Color32 = Color32::from_rgb(0xec, 0xe8, 0xf6);
pub const MUTED: Color32 = Color32::from_rgb(0x9a, 0x93, 0xb8);
pub const PEACH: Color32 = Color32::from_rgb(0xff, 0xb0, 0x8f);
pub const APRICOT: Color32 = Color32::from_rgb(0xff, 0xd3, 0xa1);
pub const VIOLET: Color32 = Color32::from_rgb(0xa9, 0x93, 0xf0);

pub fn apply(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = NIGHT;
    v.window_fill = PANEL;
    v.extreme_bg_color = Color32::from_rgb(0x12, 0x0f, 0x22);
    v.override_text_color = Some(INK);
    v.selection.bg_fill = PEACH.linear_multiply(0.45);
    v.selection.stroke = Stroke::new(1.0_f32, PEACH);
    v.hyperlink_color = VIOLET;
    for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
        w.corner_radius = CornerRadius::same(8);
    }
    v.widgets.inactive.bg_fill = PANEL;
    v.widgets.inactive.weak_bg_fill = PANEL;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, LINE);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, VIOLET);
    v.widgets.active.bg_stroke = Stroke::new(1.0_f32, PEACH);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, LINE);
    ctx.set_visuals(v);
}
