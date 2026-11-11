import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import MagicMock, patch

from brainscore_core.model_interface import TaskContext, UnifiedModel, Subject
from brainscore_core.streaming_helpers import neural_response
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
        'context_id': ['passage', 'passage'],
    }))
    stimuli.identifier = 'synthetic-language'
    return stimuli


def _language_neural_assembly():
    return NeuroidAssembly(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        coords={
            'stimulus_id': ('presentation', ['s0', 's1']),
            'object_name': ('presentation', ['sentence0', 'sentence1']),
            'sentence': ('presentation', ['the quick brown', 'fox jumps']),
            'context_id': ('presentation', ['passage', 'passage']),
            'neuroid_id': ('neuroid', ['language.0', 'language.1']),
            'layer': ('neuroid', ['language', 'language']),
        },
        dims=['presentation', 'neuroid'],
    ).reset_index('presentation')


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

    @pytest.mark.parametrize('identifier', [
        'randomembedding-100', 'randomembedding-1600', 'glove-840b',
    ])
    def test_embedding_declarations_without_a_layer_map(self, identifier):
        adapter = LanguageModelAdapter(_make_legacy_model(identifier=identifier))
        assert adapter.out_channels == {
            'neural:language_system',
            'neural:language_system_left_hemisphere',
            'neural:language_system_right_hemisphere',
        }
        assert adapter.region_layer_map == {}

    def test_explicit_output_declaration_is_copied(self):
        channels = {'neural:custom_region'}
        adapter = LanguageModelAdapter(_make_legacy_model(), out_channels=channels)
        channels.add('behavior')
        assert adapter.out_channels == {'neural:custom_region'}
        returned = adapter.out_channels
        returned.add('behavior')
        assert adapter.out_channels == {'neural:custom_region'}

    def test_empty_output_declaration_is_respected(self):
        adapter = LanguageModelAdapter(_make_legacy_model(), out_channels=set())
        assert adapter.out_channels == set()

    def test_unmapped_behavior_model_does_not_gain_neural_outputs(self):
        adapter = LanguageModelAdapter(_make_legacy_model(identifier='earley-parser'))
        assert adapter.out_channels == {'behavior'}


class TestEmbeddingSessions:

    @pytest.mark.parametrize('identifier', [
        'randomembedding-100', 'randomembedding-1600',
    ])
    @pytest.mark.parametrize('target', [
        'language_system',
        'language_system_left_hemisphere',
        'language_system_right_hemisphere',
    ])
    def test_loaded_embedding_facade_matches_legacy_exactly(
            self, identifier, target, monkeypatch):
        from brainscore_language import load_model
        from brainscore_core.legacy_facade import LegacyBenchmarkFacade

        monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'no')
        adapter = load_model(identifier)
        legacy = load_model(identifier)._legacy
        text = ['the quick brown', 'fox jumps']
        legacy.start_neural_recording(target, 'fMRI')
        expected = legacy.digest_text(text)['neural']

        facade = LegacyBenchmarkFacade(adapter)
        facade.start_neural_recording(target, 'fMRI')
        for _ in range(2):
            actual = facade.digest_text(text)['neural']
            xr.testing.assert_identical(actual, expected)
        assert adapter._legacy.neural_recordings == [(target, 'fMRI')]

    def test_loaded_embedding_typed_session_matches_legacy(self, monkeypatch):
        from brainscore_language import load_model

        monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'no')
        adapter = load_model('randomembedding-100')
        stimuli = _language_stimulus_set()
        legacy = load_model('randomembedding-100')._legacy
        legacy.start_neural_recording('language_system', 'fMRI')
        expected = legacy.digest_text(list(stimuli['sentence']))['neural']

        actual = neural_response(adapter, stimuli, record='language_system')
        np.testing.assert_array_equal(actual.values, expected.values)
        assert list(actual['stimulus_id'].values) == list(stimuli['stimulus_id'])
        assert list(actual['object_name'].values) == list(stimuli['object_name'])

    def test_embedding_rejects_behavior_before_compute(self):
        from brainscore_core.legacy_facade import LegacyBenchmarkFacade

        legacy = _make_legacy_model(identifier='randomembedding-100')
        facade = LegacyBenchmarkFacade(LanguageModelAdapter(legacy))
        with pytest.raises(NotImplementedError, match="no 'behavior' output"):
            facade.start_behavioral_task('next_word')
        legacy.start_behavioral_task.assert_not_called()
        legacy.digest_text.assert_not_called()


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

    def test_neural_response_participate_matches_legacy_process_exactly(self):
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
        scored = neural_response(
            stream_adapter, stimuli, record='language_system'
        )

        xr.testing.assert_identical(scored, legacy_output)
        legacy_stream.start_neural_recording.assert_called_once_with(
            'language_system', 'fMRI'
        )

    @pytest.mark.parametrize('channel', ['behavior', 'neural:language_system'])
    def test_typed_session_refuses_variance_before_any_legacy_call(self, channel):
        from brainscore_core.streaming_helpers import table_session
        legacy = _make_legacy_model(
            region_layer_mapping={'language_system': 'language'})
        session = table_session(_language_stimulus_set(), (channel,), task_context=TaskContext(task_type='next_word'))
        session.settings = {'require_variance': True}
        with pytest.raises(NotImplementedError, match='trial-to-trial variance'):
            LanguageModelAdapter(legacy).participate(session)
        legacy.start_behavioral_task.assert_not_called()
        legacy.start_neural_recording.assert_not_called()
        legacy.digest_text.assert_not_called()
        assert session.emitted == []


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


@pytest.mark.parametrize('defect', ['short', 'duplicate_part_number', 'wrong_stimulus'])
def test_table_rejects_invalid_row_correspondence(defect):
    stimuli = _language_stimulus_set()
    output = _language_neural_assembly()
    if defect == 'short':
        output = output.isel(presentation=[0])
    elif defect == 'duplicate_part_number':
        output = output.assign_coords(part_number=('presentation', [0, 0]))
    else:
        output = output.assign_coords(stimulus=('presentation', ['wrong', 'text']))
    legacy = _make_legacy_model()
    legacy.digest_text.return_value = {'neural': output}
    adapter = LanguageModelAdapter(legacy)
    adapter.start_recording('language_system')
    with pytest.raises(ValueError, match='presentation|part_number|stimulus coordinates'):
        adapter.process(stimuli)


def test_table_reorders_by_legacy_part_number_before_attaching_ids():
    stimuli = _language_stimulus_set()
    expected = _language_neural_assembly()
    expected.attrs['provenance'] = 'legacy-extractor'
    output = expected.assign_coords(part_number=('presentation', [0, 1])).isel(presentation=[1, 0])
    legacy = _make_legacy_model()
    legacy.digest_text.return_value = {'neural': output}
    adapter = LanguageModelAdapter(legacy)
    adapter.start_recording('language_system')
    actual = adapter.process(stimuli)
    np.testing.assert_array_equal(actual.values, expected.values)
    assert list(actual.stimulus_id.values) == ['s0', 's1']
    assert actual.attrs == expected.attrs


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

    def test_load_model_wrapped_legacy_participate_scores(self):
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

        scored = neural_response(
            model, _language_stimulus_set(), record='language_system'
        )

        assert isinstance(model, LanguageModelAdapter)
        xr.testing.assert_identical(scored, expected)

    def test_load_model_does_not_double_wrap_unified(self):
        """If the model is already a Subject, do not wrap it."""
        import brainscore_language

        class NativeSubject(Subject):
            identifier = 'native-subject'
            in_channels = set()
            out_channels = set()

            def participate(self, session):
                pass

            def reset(self):
                pass

        native = NativeSubject()

        with patch.object(brainscore_language, 'model_registry',
                          {'native-lang': lambda: native}):
            with patch('brainscore_language.import_plugin'):
                model = brainscore_language.load_model('native-lang')

        assert model is native
        assert not isinstance(model, LanguageModelAdapter)


# --- legacy benchmarks through the permanent facade -----------------------

@pytest.mark.parametrize('neural,behavior', [(True, False), (False, True), (True, True)])
def test_legacy_facade_preserves_language_outputs_in_one_call(neural, behavior):
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    legacy = _make_legacy_model(region_layer_mapping={'language_system': 'language'})
    expected = {}
    if neural:
        expected['neural'] = _language_neural_assembly()
    if behavior:
        expected['behavior'] = BehavioralAssembly(np.array([1., 2.]), dims=['presentation'])
    legacy.digest_text.return_value = expected
    facade = LegacyBenchmarkFacade(LanguageModelAdapter(legacy))
    if neural:
        facade.start_neural_recording('language_system', recording_type='spikes')
    if behavior:
        facade.start_behavioral_task('reading_times')
    text = np.array(['the quick brown', 'fox jumps'])
    actual = facade.digest_text(text)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key] is expected[key]
    assert legacy.digest_text.call_count == 1
    assert legacy.digest_text.call_args.args[0] is text
    if neural:
        legacy.start_neural_recording.assert_called_once_with('language_system', 'spikes')
    if behavior:
        legacy.start_behavioral_task.assert_called_once_with('reading_times')


def test_scoring_wraps_the_adapter_for_a_legacy_language_benchmark(monkeypatch):
    import brainscore_language
    from brainscore_core.metrics import Score

    class LegacyBenchmark:
        identifier = 'legacy-language'

        def __call__(self, candidate):
            candidate.start_neural_recording('language_system', 'fMRI')
            output = candidate.digest_text(['one sentence', 'another'])
            score = Score([1.0], coords={'aggregation': ['center']}, dims=['aggregation'])
            score.attrs.update(candidate=type(candidate).__name__, output=output)
            return score

    legacy = _make_legacy_model(region_layer_mapping={'language_system': 'language'})
    legacy.digest_text.return_value = {'neural': _language_neural_assembly()}
    monkeypatch.setattr(brainscore_language, 'import_plugin', lambda *a, **k: None)
    monkeypatch.setitem(brainscore_language.benchmark_registry, 'legacy-language', LegacyBenchmark)
    monkeypatch.setattr(brainscore_language, 'load_model', lambda _: LanguageModelAdapter(legacy))
    score = brainscore_language._run_score('m', 'legacy-language', check_mem=False)
    assert score.attrs['candidate'] == 'LegacyBenchmarkFacade'
    assert score.attrs['output']['neural'] is legacy.digest_text.return_value['neural']
    assert legacy.digest_text.call_args.args[0] == ['one sentence', 'another']
    legacy.start_neural_recording.assert_called_once_with('language_system', 'fMRI')
    assert score.attrs['compatibility']['match'] == 'full'


def test_recording_registration_is_applied_once_per_configuration():
    """Legacy subjects append registrations; a benchmark registers once, so does the adapter."""
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    legacy = _make_legacy_model(region_layer_mapping={'language_system': 'language'})
    legacy.digest_text.return_value = {'neural': _language_neural_assembly()}
    adapter = LanguageModelAdapter(legacy)
    facade = LegacyBenchmarkFacade(adapter)
    facade.start_neural_recording('language_system', recording_type='fMRI')
    facade.digest_text(['passage one'])
    facade.digest_text(['passage two'])
    legacy.start_neural_recording.assert_called_once_with('language_system', 'fMRI')
    assert legacy.digest_text.call_count == 2
    adapter.start_neural_recording('language_system', 'spikes')  # a new registration
    assert legacy.start_neural_recording.call_count == 2
    adapter.reset()
    adapter.start_neural_recording('language_system', 'fMRI')
    assert legacy.start_neural_recording.call_count == 3


def test_behavioral_task_is_applied_once_per_configuration():
    legacy = _make_legacy_model(region_layer_mapping={'language_system': 'language'})
    adapter = LanguageModelAdapter(legacy)
    adapter.start_behavioral_task('reading_times')
    adapter.start_behavioral_task('reading_times')
    legacy.start_behavioral_task.assert_called_once_with('reading_times')
    adapter.start_behavioral_task('next_word')
    assert legacy.start_behavioral_task.call_count == 2


# --- the session's requested channel selects the output ---------------------

class _AppendingSubject:
    """HuggingfaceSubject-like: registrations persist; digest_text returns every configured output."""

    def __init__(self):
        self.region_layer_mapping = {'language_system': 'transformer.h.5'}
        self.neural_recordings, self.behavioral_task = [], None
        self.digests = 0

    def identifier(self):
        return 'appending-lm'

    def start_neural_recording(self, target, recording_type):
        self.neural_recordings.append((target, recording_type))

    def start_behavioral_task(self, task):
        self.behavioral_task = task

    def digest_text(self, text):
        from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
        self.digests += 1
        text = [text] if isinstance(text, str) else list(text)
        coords = {'stimulus': ('presentation', text),
                  'part_number': ('presentation', list(range(len(text))))}
        out = {}
        if self.behavioral_task is not None:
            out['behavior'] = BehavioralAssembly(np.array(['w'] * len(text)), coords=coords,
                                                 dims=['presentation'])
        if self.neural_recordings:
            out['neural'] = NeuroidAssembly(np.zeros((len(text), 2)), dims=['presentation', 'neuroid'],
                                            coords={**coords, 'neuroid_id': ('neuroid', ['a', 'b'])})
        return out

    def reset(self):
        self.neural_recordings, self.behavioral_task = [], None


def _next_word_task():
    from brainscore_core.legacy_facade import text_stimulus_set
    stimuli = text_stimulus_set(['the cat', 'sat on'])
    return stimuli, TaskContext(task_type='next_word', metadata={'stimulus_set': stimuli})


def test_neural_then_behavior_session_returns_behavior():
    from brainscore_core.streaming_helpers import behavioral_response
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    stimuli, task = _next_word_task()
    adapter = LanguageModelAdapter(_AppendingSubject())
    assert isinstance(neural_response(adapter, stimuli, record='language_system'), NeuroidAssembly)
    assert isinstance(behavioral_response(adapter, task), BehavioralAssembly)


def test_behavior_then_neural_session_returns_neural():
    from brainscore_core.streaming_helpers import behavioral_response
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    stimuli, task = _next_word_task()
    adapter = LanguageModelAdapter(_AppendingSubject())
    assert isinstance(behavioral_response(adapter, task), BehavioralAssembly)
    assert isinstance(neural_response(adapter, stimuli, record='language_system'), NeuroidAssembly)


def test_direct_process_still_prefers_the_configured_recording():
    stimuli, _ = _next_word_task()
    legacy = _AppendingSubject()
    adapter = LanguageModelAdapter(legacy)
    adapter.start_behavioral_task('next_word')
    adapter.start_neural_recording('language_system', 'fMRI')
    assert isinstance(adapter.process(stimuli), NeuroidAssembly)


def test_typed_joint_neural_and_behavior_request_is_unchanged():
    from brainscore_core.streaming_helpers import table_session
    stimuli, task = _next_word_task()
    legacy = _AppendingSubject()
    session = table_session(stimuli, ['behavior', 'neural:language_system'], task_context=task)
    with pytest.raises(NotImplementedError):
        LanguageModelAdapter(legacy).participate(session)
    assert legacy.digests == 0


@pytest.mark.parametrize('neural,behavior', [(True, False), (False, True), (True, True)])
def test_facade_after_typed_sessions_returns_one_legacy_call(neural, behavior):
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    from brainscore_core.streaming_helpers import behavioral_response
    stimuli, task = _next_word_task()
    legacy = _AppendingSubject()
    adapter = LanguageModelAdapter(legacy)
    neural_response(adapter, stimuli, record='language_system')
    behavioral_response(adapter, task)
    facade = LegacyBenchmarkFacade(adapter)
    facade.reset()
    if neural:
        facade.start_neural_recording('language_system', recording_type='fMRI')
    if behavior:
        facade.start_behavioral_task('next_word')
    before = legacy.digests
    actual = facade.digest_text(['the cat', 'sat on'])
    assert legacy.digests == before + 1
    assert set(actual) == {k for k, on in (('neural', neural), ('behavior', behavior)) if on}
