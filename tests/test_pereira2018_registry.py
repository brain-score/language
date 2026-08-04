"""Guards on the Pereira2018 benchmark registrations.

Upstream parameterised the experiment class by metric and, in doing so, stopped
registering the linear variants in favour of ridge and linear-shuffle. The
unified-interface results are scored against ``Pereira2018.*-linear``, so those
registrations are restored on this branch. This test exists so a later merge
cannot drop them again without failing.
"""

import pytest

from brainscore_language import benchmark_registry
import brainscore_language.benchmarks.pereira2018  # noqa: F401  triggers registration

# Restored on this branch; the unified-interface scores depend on these.
LINEAR = [
    'Pereira2018.243sentences-linear',
    'Pereira2018.384sentences-linear',
    'Pereira2018.243sentences-linear-unified',
    'Pereira2018.384sentences-linear-unified',
]
# Added upstream; kept alongside rather than replaced.
UPSTREAM = [
    'Pereira2018.243sentences-ridge',
    'Pereira2018.384sentences-ridge',
    'Pereira2018.243sentences-linear-shuffle',
    'Pereira2018.384sentences-linear-shuffle',
]


@pytest.mark.parametrize('identifier', LINEAR)
def test_linear_variants_are_registered(identifier):
    assert identifier in benchmark_registry


@pytest.mark.parametrize('identifier', UPSTREAM)
def test_upstream_variants_are_registered(identifier):
    assert identifier in benchmark_registry


def test_the_two_sets_are_disjoint_and_complete():
    """Neither side was dropped when reconciling the merge."""
    registered = {k for k in benchmark_registry if k.startswith('Pereira2018')}
    assert registered == set(LINEAR) | set(UPSTREAM)


def test_linear_factories_target_the_linear_metric():
    """The restored factories must ask for the linear metric.

    They are built on the parameterised class now, where the metric is an
    argument; passing the wrong one would still construct and register, but
    would silently score a different benchmark under the linear identifier.
    """
    import inspect
    from brainscore_language.benchmarks.pereira2018 import benchmark as module

    for name in ('Pereira2018_243sentences', 'Pereira2018_384sentences'):
        source = inspect.getsource(getattr(module, name))
        assert "metric='linear_pearsonr'" in source, name
