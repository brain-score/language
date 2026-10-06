"""The public legacy scoring entry point works without modality annotations."""
from types import SimpleNamespace

import pytest

import brainscore_language as language
from brainscore_core.metrics import Score
from brainscore_core.plugin_management.conda_score import CondaScore


class LegacyBenchmark:
    identifier = 'offline-legacy'
    parent = 'behavior'

    def __call__(self, model):
        assert model.required_modalities == {'text'}
        return Score(0.5)


def test_public_score_backfills_legacy_modality(monkeypatch):
    monkeypatch.setenv('BS_INSTALL_DEPENDENCIES', 'no')
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    monkeypatch.setattr(language, 'import_plugin', lambda *a, **kw: None)
    monkeypatch.setitem(language.benchmark_registry, 'offline-legacy', LegacyBenchmark)
    model = SimpleNamespace(
        identifier='offline-model', available_modalities={'text'},
        required_modalities={'text'}, in_channels={'text'},
        required_channels={'text'}, out_channels={'behavior'}, region_layer_map={},
    )
    monkeypatch.setattr(language, 'load_model', lambda _: model)
    # Conda serialization is tested separately; do not write into the checkout.
    monkeypatch.setattr(CondaScore, 'save_score', lambda *a, **kw: None)
    result = language.score('offline-model', 'offline-legacy', check_mem=False)
    assert float(result) == 0.5
    assert result.attrs['harness_id'] == 'brainscore_language'


@pytest.mark.parametrize('error', [KeyError, FileNotFoundError])
def test_benchmark_failure_precedes_model_loading(monkeypatch, error):
    def missing(_):
        raise error('benchmark prerequisite missing')
    monkeypatch.setattr(language, 'load_benchmark', missing)
    monkeypatch.setattr(language, 'load_model', lambda _: pytest.fail('model loaded too early'))
    with pytest.raises(error, match='prerequisite'):
        language._run_score('model', 'benchmark', check_mem=False)
