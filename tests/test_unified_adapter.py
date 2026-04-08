import pytest
from unittest.mock import MagicMock, patch

from brainscore_core.model_interface import TaskContext, UnifiedModel
from brainscore_language.compat.unified_adapter import LanguageModelAdapter


def _make_legacy_model(identifier='mock-gpt2', region_layer_mapping=None,
                       identifier_is_method=True):
    """Create a mock legacy ArtificialSubject."""
    legacy = MagicMock()

    if identifier_is_method:
        # Language's identifier() is a method
        legacy.identifier = MagicMock(return_value=identifier)
    else:
        # After load_model(), identifier may be overwritten to a string
        legacy.identifier = identifier

    legacy.digest_text.return_value = {'behavior': 'mock_behavior'}
    legacy.start_behavioral_task.return_value = None
    legacy.start_neural_recording.return_value = None

    if region_layer_mapping is not None:
        legacy.region_layer_mapping = region_layer_mapping
    else:
        del legacy.region_layer_mapping

    # Remove current_tokens by default
    del legacy.current_tokens

    return legacy


class _FakeStimulusSet:
    """Minimal StimulusSet stand-in with columns and indexing."""

    def __init__(self, data, column_name='sentence'):
        self._data = data
        self._column_name = column_name
        self.columns = [column_name, 'stimulus_id']

    def __getitem__(self, key):
        if key == self._column_name:
            return MagicMock(values=self._data)
        raise KeyError(key)


class TestLanguageAdapterIsUnifiedModel:

    def test_isinstance(self):
        adapter = LanguageModelAdapter(_make_legacy_model())
        assert isinstance(adapter, UnifiedModel)


class TestLanguageAdapterIdentity:

    def test_identifier_from_method(self):
        """Language uses identifier() as a method, not a property."""
        legacy = _make_legacy_model(identifier='distilgpt2', identifier_is_method=True)
        adapter = LanguageModelAdapter(legacy)
        assert adapter.identifier == 'distilgpt2'

    def test_identifier_from_string_attribute(self):
        """After load_model(), identifier may be a string attribute."""
        legacy = _make_legacy_model(identifier='gpt2-xl', identifier_is_method=False)
        adapter = LanguageModelAdapter(legacy)
        assert adapter.identifier == 'gpt2-xl'

    def test_supported_modalities(self):
        adapter = LanguageModelAdapter(_make_legacy_model())
        assert adapter.supported_modalities == {'text'}

    def test_region_layer_map_from_attribute(self):
        legacy = _make_legacy_model(
            region_layer_mapping={'language_system': 'transformer.h.11'}
        )
        adapter = LanguageModelAdapter(legacy)
        assert adapter.region_layer_map == {'language_system': 'transformer.h.11'}

    def test_region_layer_map_empty_when_no_mapping(self):
        legacy = _make_legacy_model()
        adapter = LanguageModelAdapter(legacy)
        assert adapter.region_layer_map == {}


class TestLanguageAdapterProcess:

    def test_process_extracts_sentence_column(self):
        legacy = _make_legacy_model()
        legacy.digest_text.return_value = {'behavior': 'behavioral_assembly'}
        adapter = LanguageModelAdapter(legacy)
        adapter.start_task(TaskContext(task_type='next_word'))
        stimuli = _FakeStimulusSet(['the quick brown', 'fox jumps'])

        result = adapter.process(stimuli)

        legacy.digest_text.assert_called_once_with(['the quick brown', 'fox jumps'])
        assert result == 'behavioral_assembly'

    def test_process_extracts_text_column(self):
        legacy = _make_legacy_model()
        legacy.digest_text.return_value = {'behavior': 'behavioral_assembly'}
        adapter = LanguageModelAdapter(legacy)
        adapter.start_task(TaskContext(task_type='next_word'))
        stimuli = _FakeStimulusSet(['hello world'], column_name='text')

        result = adapter.process(stimuli)

        legacy.digest_text.assert_called_once_with(['hello world'])

    def test_process_passes_raw_text_when_no_columns(self):
        legacy = _make_legacy_model()
        legacy.digest_text.return_value = {'behavior': 'behavioral_assembly'}
        adapter = LanguageModelAdapter(legacy)
        adapter.start_task(TaskContext(task_type='next_word'))

        result = adapter.process(['the quick brown'])

        legacy.digest_text.assert_called_once_with(['the quick brown'])

    def test_process_returns_identical_output_to_digest_text_behavioral(self):
        """Core requirement: process() extracts the right assembly from digest_text dict."""
        legacy = _make_legacy_model()
        sentinel = object()
        legacy.digest_text.return_value = {'behavior': sentinel}
        adapter = LanguageModelAdapter(legacy)
        adapter.start_task(TaskContext(task_type='next_word'))

        result = adapter.process(['text'])
        assert result is sentinel

    def test_process_returns_neural_when_recording(self):
        legacy = _make_legacy_model()
        neural_sentinel = object()
        legacy.digest_text.return_value = {'neural': neural_sentinel}
        adapter = LanguageModelAdapter(legacy)
        adapter.start_recording('language_system')

        result = adapter.process(['text'])
        assert result is neural_sentinel

    def test_process_prefers_neural_when_both_active(self):
        """When both task and recording are active, neural takes precedence."""
        legacy = _make_legacy_model()
        neural = object()
        behavior = object()
        legacy.digest_text.return_value = {'neural': neural, 'behavior': behavior}
        adapter = LanguageModelAdapter(legacy)
        adapter.start_recording('language_system')
        adapter.start_task(TaskContext(task_type='next_word'))

        result = adapter.process(['text'])
        assert result is neural

    def test_process_single_key_fallback(self):
        """When only one key in dict and no mode set, return that value."""
        legacy = _make_legacy_model()
        sentinel = object()
        legacy.digest_text.return_value = {'behavior': sentinel}
        adapter = LanguageModelAdapter(legacy)

        result = adapter.process(['text'])
        assert result is sentinel


class TestLanguageAdapterStartTask:

    def test_start_task_unwraps_to_single_arg(self):
        """Legacy start_behavioral_task(task) takes ONE arg, unlike vision's two."""
        legacy = _make_legacy_model()
        adapter = LanguageModelAdapter(legacy)
        ctx = TaskContext(task_type='next_word', label_set=['a', 'b'])

        adapter.start_task(ctx)

        legacy.start_behavioral_task.assert_called_once_with('next_word')


class TestLanguageAdapterStartRecording:

    def test_start_recording_with_explicit_type(self):
        legacy = _make_legacy_model()
        adapter = LanguageModelAdapter(legacy)

        adapter.start_recording('language_system', recording_type='ECoG')

        legacy.start_neural_recording.assert_called_once_with('language_system', 'ECoG')

    def test_start_recording_default_fmri(self):
        """Default recording_type for language is 'fMRI'."""
        legacy = _make_legacy_model()
        adapter = LanguageModelAdapter(legacy)

        adapter.start_recording('language_system')

        legacy.start_neural_recording.assert_called_once_with('language_system', 'fMRI')


class TestLanguageAdapterReset:

    def test_reset_clears_state(self):
        legacy = _make_legacy_model()
        adapter = LanguageModelAdapter(legacy)
        adapter.start_task(TaskContext(task_type='next_word'))
        adapter.start_recording('language_system')

        adapter.reset()

        assert adapter._task_context is None
        assert adapter._recording_active is False
        assert adapter._task_active is False

    def test_reset_clears_current_tokens(self):
        legacy = _make_legacy_model()
        # Re-add current_tokens for this test
        legacy.current_tokens = ['some', 'tokens']
        adapter = LanguageModelAdapter(legacy)

        adapter.reset()

        assert legacy.current_tokens is None
