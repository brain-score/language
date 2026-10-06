"""Keep Pereira identifiers, metrics and evaluation routes consistent."""
from unittest.mock import Mock

import pytest

from brainscore_language import benchmark_registry
from brainscore_language.benchmarks.pereira2018 import benchmark, unified

VARIANTS = {
    'linear': ('linear_pearsonr', False),
    'linear-unified': ('linear_pearsonr', True),
    'ridge': ('ridge_pearsonr', False),
    'ridge-unified': ('ridge_pearsonr', True),
    'linear-shuffle': ('linear_pearsonr', False),
}


def test_registry_contains_all_ten_variants():
    expected = {
        f'Pereira2018.{experiment}-{variant}'
        for experiment in ('243sentences', '384sentences')
        for variant in VARIANTS
    }
    registered = {key for key in benchmark_registry if key.startswith('Pereira2018')}
    assert registered == expected


@pytest.mark.parametrize('experiment', ['243sentences', '384sentences'])
@pytest.mark.parametrize('variant', VARIANTS)
def test_registered_factory_selects_metric_and_route(monkeypatch, experiment, variant):
    # Replace construction to inspect configuration without downloading brain data.
    legacy_constructor, unified_constructor = Mock(), Mock()
    monkeypatch.setattr(benchmark, '_Pereira2018Experiment', legacy_constructor)
    monkeypatch.setattr(unified, '_Pereira2018ExperimentUnified', unified_constructor)
    metric, is_unified = VARIANTS[variant]
    selected, unused = ((unified_constructor, legacy_constructor) if is_unified
                        else (legacy_constructor, unified_constructor))

    result = benchmark_registry[f'Pereira2018.{experiment}-{variant}']()

    selected.assert_called_once()
    unused.assert_not_called()
    assert result is selected.return_value
    kwargs = selected.call_args.kwargs
    assert kwargs['experiment'] == experiment
    assert kwargs['metric'] == metric
    if variant.startswith('ridge'):
        assert kwargs['crossvalidation_kwargs'] == {
            'split_coord': 'story', 'kfold': 'group', 'random_state': 1234,
        }
    elif variant == 'linear-shuffle':
        assert kwargs['identifier_suffix'] == '-shuffle'
        assert kwargs['crossvalidation_kwargs'] == {
            'splits': 10, 'train_size': 0.9, 'kfold': False, 'random_state': 1,
        }
    else:
        assert 'crossvalidation_kwargs' not in kwargs
