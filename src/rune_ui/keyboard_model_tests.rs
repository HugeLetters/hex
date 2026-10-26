use super::*;
use rune::text::TextSystem;
use rune::{Env, Runtime};

fn runtime(scene: &str) -> Runtime<Ui> {
    let mut state = Ui::with(AppSettings::default(), Service::Fixture(Vec::new()));
    SCENES.iter().find(|(name, _)| *name == scene).unwrap().1(&mut state);
    if scene == "portal" {
        state.platform.caps = platform::Capabilities::LINUX;
    }
    let text = TextSystem::new(rune_theme_opencode::FONTS, rune_theme_opencode::FAMILIES);
    let mut rt = Runtime::with_text(
        state,
        view,
        (980.0, 1720.0),
        Env {
            system_dark: true,
            scale: 1.0,
        },
        text,
    );
    rt.settle_all();
    rt.frame(0.0);
    rt
}

fn tap(rt: &mut Runtime<Ui>, key: &str) {
    let (x, y) = rt.find(key).unwrap_or_else(|| panic!("no element {key}"));
    rt.pointer_down(x, y);
    rt.pointer_up(x, y);
    rt.settle_all();
    rt.frame(0.0);
}

#[test]
fn keyboard_model_switching_is_a_macos_only_saved_opt_in() {
    let mut rt = runtime("mac-settings");
    assert!(!rt.state.settings.keyboard_model_switching.enabled);
    tap(&mut rt, "keyboard-model-switching");
    assert!(rt.state.settings.keyboard_model_switching.enabled);
    assert_eq!(rt.state.saved, 1);
    tap(&mut rt, "keyboard-model-switching");
    assert!(!rt.state.settings.keyboard_model_switching.enabled);
    for scene in ["portal", "windows-settings"] {
        let rt = runtime(scene);
        assert!(rt.find("keyboard-model-switching").is_none());
    }
}

#[test]
fn keyboard_model_switching_maps_enabled_sources_to_installed_models_or_no_switch() {
    let mut rt = runtime("mac-settings");
    rt.state.platform.input_sources = vec![crate::keyboard_model_switching::InputSource {
        id: "com.apple.keylayout.Spanish".into(),
        name: "Spanish".into(),
    }];
    rt.state.picker.installed = Some(vec![TranscriptionModelId::ParakeetV3]);
    rt.invalidate();
    rt.frame(0.0);
    tap(&mut rt, "keyboard-model-switching");
    tap(&mut rt, "input-source-model0-trigger");
    tap(&mut rt, "input-source-model0-item1");
    assert_eq!(
        rt.state
            .settings
            .keyboard_model_switching
            .mappings
            .get("com.apple.keylayout.Spanish"),
        Some(&TranscriptionModelId::ParakeetV3)
    );
    tap(&mut rt, "input-source-model0-trigger");
    tap(&mut rt, "input-source-model0-item0");
    assert!(
        rt.state
            .settings
            .keyboard_model_switching
            .mappings
            .is_empty()
    );
}
