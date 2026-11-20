"""Keep Pereira identifiers, metrics and evaluation routes consistent."""
from unittest.mock import Mock

import pytest

from brainscore_language import benchmark_registry
from brainscore_language.benchmarks.pereira2018 import benchmark

VARIANTS = {
    'ridge': 'ridge_pearsonr',
    'linear-shuffle': 'linear_pearsonr',
}


def test_registry_contains_the_four_variants():
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
    constructor = Mock()
    monkeypatch.setattr(benchmark, '_Pereira2018Experiment', constructor)
    metric = VARIANTS[variant]

    result = benchmark_registry[f'Pereira2018.{experiment}-{variant}']()

    constructor.assert_called_once()
    assert result is constructor.return_value
    kwargs = constructor.call_args.kwargs
    assert kwargs['experiment'] == experiment
    assert kwargs['metric'] == metric
    if variant == 'ridge':
        assert kwargs['crossvalidation_kwargs'] == {
            'split_coord': 'story', 'kfold': 'group', 'random_state': 1234,
        }
    else:
        assert kwargs['identifier_suffix'] == '-shuffle'
        assert kwargs['crossvalidation_kwargs'] == {
            'splits': 10, 'train_size': 0.9, 'kfold': False, 'random_state': 1,
        }


@pytest.mark.parametrize('lazy', [False, True])
def test_ceiling_raw_check_accepts_lazy_arrays(monkeypatch, lazy):
    """Exercise the plugin's actual ceiling assertion without downloading data."""
    import numpy as np
    import dask.array as da
    from types import SimpleNamespace
    from brainscore_core.metrics import Score
    from brainscore_language.benchmarks.pereira2018 import test as plugin_tests

    values = np.array([0.2, 0.4, np.nan, 0.6])
    raw = Score(da.from_array(values, chunks=2) if lazy else values, dims=['neuroid'])
    raw.attrs['raw'] = Score(
        np.zeros((1, 1, 1, 4)), dims=['sub_subject', 'num_subjects', 'split', 'neuroid'])
    ceiling = Score(0.4)
    ceiling.attrs['raw'] = raw
    monkeypatch.setattr(plugin_tests, 'load_benchmark', lambda _: SimpleNamespace(ceiling=ceiling))
    plugin_tests.TestBenchmark().test_ceiling_raw(243)
