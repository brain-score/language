import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import MagicMock, patch

from brainscore_core.model_interface import TaskContext, UnifiedModel, BrainScoreModel
from brainscore_core.streaming_helpers import score_stimuli
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
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


def _language_stimulus_set():
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': ['s0', 's1'],
        'sentence': ['the quick brown', 'fox jumps'],
        'object_name': ['sentence0', 'sentence1'],
    }))
    stimuli.identifier = 'synthetic-language'
    return stimuli


def _language_neural_assembly():
    return NeuroidAssembly(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        coords={
            'stimulus_id': ('presentation', ['s0', 's1']),
            'object_name': ('presentation', ['sentence0', 'sentence1']),
            'neuroid_id': ('neuroid', ['language.0', 'language.1']),
            'layer': ('neuroid', ['language', 'language']),
        },
        dims=['presentation', 'neuroid'],
    )


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

    def test_score_stimuli_interact_matches_legacy_process_exactly(self):
        stimuli = _language_stimulus_set()
        expected = _language_neural_assembly()

        legacy_expected = _make_legacy_model(
            region_layer_mapping={'language_system': 'language'}
        )
        legacy_expected.digest_text.return_value = {'neural': expected}
        expected_adapter = LanguageModelAdapter(legacy_expected)
        expected_adapter.start_recording('language_system')
        legacy_output = expected_adapter.process(stimuli)

        legacy_stream = _make_legacy_model(
            region_layer_mapping={'language_system': 'language'}
        )
        legacy_stream.digest_text.return_value = {'neural': expected}
        stream_adapter = LanguageModelAdapter(legacy_stream)
        scored = score_stimuli(
            stream_adapter, stimuli, record='language_system'
        )

        xr.testing.assert_identical(scored, legacy_output)
        legacy_stream.start_neural_recording.assert_called_once_with(
            'language_system', 'fMRI'
        )


class TestLanguageAdapterLegacyMethods:

    def test_digest_text_delegates(self):
        """Existing benchmarks call digest_text() directly on the adapter."""
        legacy = _make_legacy_model()
        result_dict = {'behavior': 'behavioral_assembly'}
        legacy.digest_text.return_value = result_dict
        adapter = LanguageModelAdapter(legacy)

        result = adapter.digest_text(['the quick brown'])

        assert result is result_dict
        legacy.digest_text.assert_called_once_with(['the quick brown'])

    def test_start_behavioral_task_delegates(self):
        """Existing benchmarks call start_behavioral_task() directly."""
        legacy = _make_legacy_model()
        adapter = LanguageModelAdapter(legacy)

        adapter.start_behavioral_task('next_word')

        legacy.start_behavioral_task.assert_called_once_with('next_word')
        assert adapter._task_active is True

    def test_start_neural_recording_delegates(self):
        """Existing benchmarks call start_neural_recording() directly."""
        legacy = _make_legacy_model()
        adapter = LanguageModelAdapter(legacy)

        adapter.start_neural_recording('language_system', 'fMRI')

        legacy.start_neural_recording.assert_called_once_with('language_system', 'fMRI')
        assert adapter._recording_active is True


class TestLanguageAdapterStartTask:

    def test_start_task_unwraps_to_single_arg(self):
        """New API: start_task(TaskContext(...))."""
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


class TestLanguageAutoWrapping:

    def test_load_model_wraps_legacy(self):
        """load_model() should wrap a legacy ArtificialSubject in LanguageModelAdapter."""
        import brainscore_language
        legacy = _make_legacy_model(identifier='test-gpt2', identifier_is_method=False)

        with patch.object(brainscore_language, 'model_registry',
                          {'test-gpt2': lambda: legacy}):
            with patch('brainscore_language.import_plugin'):
                model = brainscore_language.load_model('test-gpt2')

        assert isinstance(model, UnifiedModel)
        assert isinstance(model, LanguageModelAdapter)
        # After load_model, identifier was overwritten to string 'test-gpt2'
        assert model.identifier == 'test-gpt2'

    def test_load_model_wrapped_legacy_interact_scores(self):
        import brainscore_language
        expected = _language_neural_assembly()
        legacy = _make_legacy_model(
            identifier='test-gpt2',
            identifier_is_method=False,
            region_layer_mapping={'language_system': 'language'},
        )
        legacy.digest_text.return_value = {'neural': expected}

        with patch.object(brainscore_language, 'model_registry',
                          {'test-gpt2': lambda: legacy}):
            with patch('brainscore_language.import_plugin'):
                model = brainscore_language.load_model('test-gpt2')

        scored = score_stimuli(
            model, _language_stimulus_set(), record='language_system'
        )

        assert isinstance(model, LanguageModelAdapter)
        xr.testing.assert_identical(scored, expected)

    def test_load_model_does_not_double_wrap_unified(self):
        """If the model is already a UnifiedModel, don't wrap it."""
        import brainscore_language

        native = BrainScoreModel(
            identifier='native-lang',
            model=None,
            region_layer_map={},
            preprocessors={'text': lambda m, s, **kw: None},
        )

        with patch.object(brainscore_language, 'model_registry',
                          {'native-lang': lambda: native}):
            with patch('brainscore_language.import_plugin'):
                model = brainscore_language.load_model('native-lang')

        assert model is native
        assert not isinstance(model, LanguageModelAdapter)
