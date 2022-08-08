import numpy as np
from numpy.random import RandomState
from pytest import approx

from brainio.assemblies import NeuroidAssembly
from brainscore_language import load_dataset, load_metric
from brainscore_language.plugins.schrimpf2021.metric import linear_regression


class TestData:
    def test_Pereira2018language(self):
        assembly = load_dataset('Pereira2018.language_system')
        assert set(assembly['experiment'].values) == {'243sentences', '384sentences'}
        assert len(assembly['presentation']) == 243 + 384
        assert len(set(assembly['subject'].values)) == 10
        assert len(set(assembly['neuroid_id'].values)) == 13553
        assert np.nansum(assembly.values) == approx(1935595.263162177)

    def test_Pereira2018auditory(self):
        assembly = load_dataset('Pereira2018.auditory')
        assert set(assembly['experiment'].values) == {'243sentences', '384sentences'}
        assert len(assembly['presentation']) == 243 + 384
        assert len(set(assembly['subject'].values)) == 10
        assert len(set(assembly['neuroid_id'].values)) == 5692
        assert np.nansum(assembly.values) == approx(-257124.1144940494)

    def test_Fedorenko2016(self):
        assembly = load_dataset('Fedorenko2016.language')
        assert len(assembly['presentation']) == 416
        assert len(assembly['neuroid']) == 97
        assert len(np.unique(assembly['subject_UID'])) == 5

    def test_Blank2014(self):
        assembly = load_dataset('Blank2014.fROI')
        assert len(assembly['presentation']) == 1317
        assert len(assembly['neuroid']) == 60
        assert len(set(assembly['stimulus_id'].values)) == len(assembly['presentation'])
        assert set(assembly['story'].values) == {'Aqua', 'Boar', 'Elvis', 'HighSchool',
                                                 'KingOfBirds', 'MatchstickSeller', 'MrSticky', 'Tulips'}
        assert set(assembly['subject_id'].values) == {'090', '061', '085', '088', '098'}
        assert set(assembly['fROI_area'].values) == {'10_RH_IFGorb', '11_RH_MFG', '03_LH_IFG', '09_RH_IFG',
                                                     '01_LH_PostTemp', '12_RH_AngG', '07_RH_PostTemp', '04_LH_IFGorb',
                                                     '06_LH_AngG', '02_LH_AntTemp', '08_RH_AntTemp', '05_LH_MFG'}

        mean_assembly = assembly.groupby('subject_id').mean()
        assert not np.isnan(mean_assembly).any()


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
        metric = load_metric('linear_predictivity')
        score = metric(source=assembly, target=assembly)
        assert score.sel(aggregation='center') == approx(1, abs=.00001)

    def test_offset_source_target(self):
        source = self._make_assembly()
        target = source + 2  # offset all values
        metric = load_metric('linear_predictivity')
        score = metric(source=source, target=target)
        assert score.sel(aggregation='center') == approx(1, abs=.00001)

    def test_mismatched_source_target(self):
        random_state = RandomState(1)
        source = (np.arange(30 * 25) + random_state.standard_normal(30 * 25)).reshape((30, 25))
        source = self._make_assembly(source)
        target = random_state.poisson(lam=1, size=30 * 25).reshape((30, 25))
        target = self._make_assembly(target)
        metric = load_metric('linear_predictivity')
        score = metric(source=source, target=target)
        assert score.sel(aggregation='center') == approx(.02826294, abs=.00001)

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
