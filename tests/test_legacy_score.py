"""The public legacy scoring entry point works without modality annotations."""

import pytest

import brainscore_language as language
from brainscore_core.contract import Subject
from brainscore_core.metrics import Score
from brainscore_core.plugin_management.conda_score import CondaScore


class LegacyBenchmark:
    identifier = 'offline-legacy'
    parent = 'behavior'

    def __call__(self, model):
        # The benchmark receives the legacy facade; the subject is behind it.
        assert model.subject.required_channels == {"text"}
        return Score(0.5)


def test_public_score_backfills_legacy_modality(monkeypatch):
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'no')
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    monkeypatch.setattr(language, 'import_plugin', lambda *a, **kw: None)
    monkeypatch.setitem(language.benchmark_registry, 'offline-legacy', LegacyBenchmark)
    class Offline(Subject):
        identifier = 'offline-model'
        in_channels = required_channels = {'text'}
        out_channels = {'behavior'}
        region_layer_map = {}

        def participate(self, session):
            pass

    model = Offline()
    monkeypatch.setattr(language, 'load_model', lambda _: model)
    # Conda serialization is tested separately; do not write into the checkout.
    monkeypatch.setattr(CondaScore, 'save_score', lambda *a, **kw: None)
    result = language.score('offline-model', 'offline-legacy', check_mem=False)
    assert float(result) == 0.5
    assert result.attrs['entry_point'] == 'brainscore_language'


@pytest.mark.parametrize('error', [KeyError, FileNotFoundError])
def test_benchmark_failure_precedes_model_loading(monkeypatch, error):
    def missing(_):
        raise error('benchmark prerequisite missing')
    monkeypatch.setattr(language, 'load_benchmark', missing)
    monkeypatch.setattr(language, 'load_model', lambda _: pytest.fail('model loaded too early'))
    with pytest.raises(error, match='prerequisite'):
        language._run_score('model', 'benchmark', check_mem=False)


class _LabelSubject(Subject):
    """Participate-only subject that answers behavior:choice; the last n_outside rows are invalid."""
    identifier = 'label-subject'
    in_channels = {'text'}
    out_channels = {'behavior:choice'}
    required_channels = set()
    region_layer_map = {}

    def __init__(self, n_outside):
        self.n_outside = n_outside

    def participate(self, session):
        import numpy as np
        from brainscore_core.streaming import StreamEvent
        from brainscore_core.response_formats import response_record
        from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
        while session.next_input() is not None:
            pass
        ids = list(session.stimuli['stimulus_id'])
        rows = np.zeros((len(ids), 2), dtype=np.float32)
        rows[:, 0] = 1.0
        if self.n_outside:
            rows[-self.n_outside:] = np.nan
        session.respond(StreamEvent('behavior:choice', BehavioralAssembly(
            rows, coords={'stimulus_id': ('presentation', ids), 'choice': ['cat', 'dog']},
            dims=['presentation', 'choice'],
            attrs={'response': response_record('behavior:choice', 'test')}), None))

    def reset(self):
        pass


class LabelBenchmark:
    identifier = 'offline-label'
    parent = 'behavior'

    def __call__(self, model):
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
        stimuli = StimulusSet(pd.DataFrame({'stimulus_id': [f's{i}' for i in range(20)],
                                            'sentence': ['word'] * 20}))
        stimuli.identifier = 'language-entry-label'
        model.start_task('label', ['cat', 'dog'])
        model.look_at(stimuli)
        return Score(0.5)


def _label_score(monkeypatch, n_outside):
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'no')
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    monkeypatch.setattr(language, 'import_plugin', lambda *a, **kw: None)
    monkeypatch.setitem(language.benchmark_registry, 'offline-label', LabelBenchmark)
    monkeypatch.setattr(language, 'load_model', lambda _: _LabelSubject(n_outside))
    monkeypatch.setattr(CondaScore, 'save_score', lambda *a, **kw: None)
    return language.score('label-subject', 'offline-label', check_mem=False)


def test_the_score_carries_the_replaced_answer_count(monkeypatch):
    with pytest.warns(UserWarning, match='1 invalid answers'):
        result = _label_score(monkeypatch, 1)
    assert result.attrs['n_invalid'] == 1
    assert 'random label' in result.attrs['answer_policy']


def test_the_score_carries_a_zero_when_no_answer_was_replaced(monkeypatch):
    result = _label_score(monkeypatch, 0)
    assert result.attrs['n_invalid'] == 0
    assert 'random label' in result.attrs['answer_policy']
