import numpy as np
from numpy.random import RandomState
from pytest import approx

from brainio.assemblies import NeuroidAssembly
from brainscore_language import load_metric
from brainscore_language.metrics.linear_predictivity import linear_regression


class TestMetric:
    def test_regression_identical(self):
        assembly = self._make_assembly()
        regression = linear_regression()
        regression.fit(source=assembly, target=assembly)
        prediction = regression.predict(source=assembly)
        assert all(prediction['stimulus_id'] == assembly['stimulus_id'])
        assert all(prediction['neuroid_id'] == assembly['neuroid_id'])

    def test_identical_source_target(self):
        assembly = self._make_assembly()
        metric = load_metric('linear_pearsonr')
        score = metric(assembly1=assembly, assembly2=assembly)
        assert score == approx(1, abs=.00001)

    def test_offset_source_target(self):
        source = self._make_assembly()
        target = source + 2  # offset all values
        metric = load_metric('linear_pearsonr')
        score = metric(assembly1=source, assembly2=target)
        assert score == approx(1, abs=.00001)

    def test_mismatched_source_target(self):
        random_state = RandomState(1)
        source = (np.arange(30 * 25) + random_state.standard_normal(30 * 25)).reshape((30, 25))
        source = self._make_assembly(source)
        target = random_state.poisson(lam=1, size=30 * 25).reshape((30, 25))
        target = self._make_assembly(target)
        metric = load_metric('linear_pearsonr')
        score = metric(assembly1=source, assembly2=target)
        assert score == approx(.02826294, abs=.00001)

    def test_weights_stored(self):
        assembly = self._make_assembly()
        metric = load_metric('linear_pearsonr', store_regression_weights=True)
        score = metric(assembly1=assembly, assembly2=assembly)
        assert score.attrs['raw_regression_coef'].shape == (10, 25, 25), \
            "Should be 10 splits x 25 source neuroids x 25 target neuroids"
        assert score.attrs['raw_regression_coef'].dims == ('split', 'source_neuroid', 'target_neuroid')
        assert score.attrs['raw_regression_intercept'].shape == (10, 25), \
            "should be 10 splits x 25 target neuroids"
        assert score.attrs['raw_regression_intercept'].dims == ('split', 'target_neuroid')

    def _make_assembly(self, values=None):
        if values is None:
            values = RandomState(1).standard_normal(30 * 25).reshape((30, 25))
        assembly = NeuroidAssembly(values,
                                   coords={'stimulus_id': ('presentation', np.arange(30)),
                                           'stimulus_category': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', ['some_region'] * 25)},
                                   dims=['presentation', 'neuroid'])
        return assembly
