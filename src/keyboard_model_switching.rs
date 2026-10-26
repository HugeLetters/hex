//! Input-source changes are model-picker choices, not recording-time routing.

use std::time::{Duration, Instant};

use crate::app_settings::KeyboardModelSwitching;
use crate::transcription_models::TranscriptionModelId;

const DEBOUNCE: Duration = Duration::from_millis(300);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InputSource {
    pub(crate) id: String,
    pub(crate) name: String,
}

/// No models or workers live here: this only admits the latest picker choice.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct Switcher {
    settings: KeyboardModelSwitching,
    current: Option<String>,
    due: Option<Instant>,
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
impl Switcher {
    pub(crate) fn new(settings: KeyboardModelSwitching, current: Option<String>) -> Self {
        Self {
            settings,
            current,
            due: None,
        }
    }

    /// Returns true on a real change, so the owner can cancel obsolete work.
    pub(crate) fn source_changed(&mut self, source: Option<String>, now: Instant) -> bool {
        if self.current == source {
            return false;
        }
        self.current = source;
        self.due = self.mapped_model().map(|_| now + DEBOUNCE);
        true
    }

    pub(crate) fn mapped_model(&self) -> Option<TranscriptionModelId> {
        self.settings
            .enabled
            .then(|| self.settings.mappings.get(self.current.as_ref()?))
            .flatten()
            .copied()
    }

    pub(crate) fn take_due(&mut self, now: Instant) -> Option<TranscriptionModelId> {
        if now < self.due? {
            return None;
        }
        self.due = None;
        self.mapped_model()
    }

    pub(crate) fn manual_choice(&mut self) {
        self.due = None;
    }

    /// Only enabling, disabling, or editing the active mapping affects work.
    /// Returns true when the owner must supersede an automatic preparation.
    pub(crate) fn configure(&mut self, settings: KeyboardModelSwitching, now: Instant) -> bool {
        let before = (self.settings.enabled, self.mapped_model());
        self.settings = settings;
        if before == (self.settings.enabled, self.mapped_model()) {
            return false;
        }
        self.due = self.mapped_model().map(|_| now);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_settings::KeyboardModelSwitching;
    use crate::transcription_models::TranscriptionModelId;
    use std::time::{Duration, Instant};

    #[test]
    fn a_mapped_startup_source_is_known_before_any_settings_edit_but_does_not_switch() {
        let now = Instant::now();
        let mut settings = KeyboardModelSwitching {
            enabled: true,
            mappings: [("Russian".into(), TranscriptionModelId::ParakeetV3)].into(),
        };
        let mut switcher = Switcher::new(settings.clone(), Some("Russian".into()));
        assert_eq!(switcher.take_due(now + Duration::from_secs(1)), None);
        settings
            .mappings
            .insert("Russian".into(), TranscriptionModelId::WhisperLargeV3Turbo);
        assert!(switcher.configure(settings, now));
        assert_eq!(
            switcher.take_due(now),
            Some(TranscriptionModelId::WhisperLargeV3Turbo)
        );
    }

    #[test]
    fn startup_preserves_manual_selection_and_source_changes_are_debounced() {
        let settings = KeyboardModelSwitching {
            enabled: true,
            mappings: [("Russian".into(), TranscriptionModelId::ParakeetV3)].into(),
        };
        let now = Instant::now();
        let mut switcher = Switcher::new(settings, Some("ABC".into()));
        assert!(!switcher.source_changed(Some("ABC".into()), now));
        assert_eq!(switcher.take_due(now + Duration::from_secs(1)), None);
        assert!(switcher.source_changed(Some("Russian".into()), now));
        assert_eq!(switcher.take_due(now + Duration::from_millis(299)), None);
        assert_eq!(
            switcher.take_due(now + Duration::from_millis(300)),
            Some(TranscriptionModelId::ParakeetV3)
        );
        assert_eq!(switcher.take_due(now + Duration::from_secs(1)), None);
    }

    #[test]
    fn a_manual_choice_cancels_the_pending_switch_until_another_source_change() {
        let now = Instant::now();
        let mut switcher = Switcher::new(
            KeyboardModelSwitching {
                enabled: true,
                mappings: [("Russian".into(), TranscriptionModelId::ParakeetV3)].into(),
            },
            Some("ABC".into()),
        );
        switcher.source_changed(Some("ABC".into()), now);
        switcher.source_changed(Some("Russian".into()), now);
        switcher.manual_choice();
        assert_eq!(switcher.take_due(now + Duration::from_secs(1)), None);
        // Duplicate native notifications must not undo the manual choice.
        assert!(!switcher.source_changed(Some("Russian".into()), now));
        assert_eq!(switcher.take_due(now + Duration::from_secs(2)), None);
    }

    #[test]
    fn enabling_or_editing_the_active_mapping_applies_now_but_other_edits_do_not() {
        let now = Instant::now();
        let mut switcher = Switcher::new(KeyboardModelSwitching::default(), Some("Russian".into()));
        switcher.source_changed(Some("Russian".into()), now);
        let mut settings = KeyboardModelSwitching {
            enabled: true,
            mappings: [("Russian".into(), TranscriptionModelId::ParakeetV3)].into(),
        };
        assert!(switcher.configure(settings.clone(), now));
        assert_eq!(
            switcher.take_due(now),
            Some(TranscriptionModelId::ParakeetV3)
        );
        settings
            .mappings
            .insert("ABC".into(), TranscriptionModelId::ParakeetUnifiedEnglish);
        assert!(!switcher.configure(settings.clone(), now));
        assert_eq!(switcher.take_due(now), None);
        settings
            .mappings
            .insert("Russian".into(), TranscriptionModelId::WhisperLargeV3Turbo);
        assert!(switcher.configure(settings.clone(), now));
        assert_eq!(
            switcher.take_due(now),
            Some(TranscriptionModelId::WhisperLargeV3Turbo)
        );
        settings.enabled = false;
        assert!(switcher.configure(settings, now));
        assert_eq!(switcher.take_due(now), None);
    }

    #[test]
    fn cycling_past_a_source_only_switches_to_the_last_mapping() {
        let now = Instant::now();
        let mut switcher = Switcher::new(
            KeyboardModelSwitching {
                enabled: true,
                mappings: [
                    ("Spanish".into(), TranscriptionModelId::WhisperLargeV3Turbo),
                    ("Russian".into(), TranscriptionModelId::ParakeetV3),
                ]
                .into(),
            },
            Some("ABC".into()),
        );
        switcher.source_changed(Some("ABC".into()), now);
        switcher.source_changed(Some("Spanish".into()), now);
        switcher.source_changed(Some("Russian".into()), now + Duration::from_millis(100));
        assert_eq!(switcher.take_due(now + Duration::from_millis(300)), None);
        assert_eq!(
            switcher.take_due(now + Duration::from_millis(400)),
            Some(TranscriptionModelId::ParakeetV3)
        );
        switcher.source_changed(Some("Spanish".into()), now + Duration::from_secs(1));
        switcher.source_changed(Some("ABC".into()), now + Duration::from_millis(1100));
        assert_eq!(switcher.take_due(now + Duration::from_secs(2)), None);
    }

    #[test]
    fn turning_off_the_feature_cancels_a_debounced_switch() {
        let now = Instant::now();
        let mut settings = KeyboardModelSwitching {
            enabled: true,
            mappings: [("Russian".into(), TranscriptionModelId::ParakeetV3)].into(),
        };
        let mut switcher = Switcher::new(settings.clone(), Some("ABC".into()));
        switcher.source_changed(Some("ABC".into()), now);
        switcher.source_changed(Some("Russian".into()), now);
        settings.enabled = false;
        switcher.configure(settings, now + Duration::from_millis(100));
        assert_eq!(switcher.take_due(now + Duration::from_secs(1)), None);
        switcher.source_changed(Some("ABC".into()), now);
        switcher.source_changed(Some("Russian".into()), now);
        assert_eq!(switcher.take_due(now + Duration::from_secs(2)), None);
    }
}
