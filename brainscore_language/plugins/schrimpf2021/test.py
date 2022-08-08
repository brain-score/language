import numpy as np
from numpy.random import RandomState
from pytest import approx

from brainio.assemblies import NeuroidAssembly
from brainscore_language import load_dataset, load_metric, ArtificialSubject, load_benchmark
from brainscore_language.plugins.schrimpf2021.metric import linear_regression


class TestData:
    def test_Pereira2018(self):
        assembly = load_dataset('Pereira2018.language_system')
        assert set(assembly['experiment'].values) == {'243sentences', '384sentences'}
        assert len(assembly['presentation']) == 243 + 384
        assert len(set(assembly['stimulu'].values)) == 243 + 384
        assert 'Once upon a time' in assembly['stimuli'].values
        assert len(set(assembly['subject'].values)) == 10
        assert len(set(assembly['neuroid_id'].values)) == 13553
        assert np.nansum(assembly.values) == approx(1935595.263162177)

    def test_Fedorenko2016(self):
        assembly = load_dataset('Fedorenko2016.language')
        assert len(assembly['presentation']) == 416
        assert len(set(assembly['stimulus'].values)) == len(set(assembly['word'].values)) == 255
        assert ' '.join(assembly.sel(sentence_id=0)['stimulus'].values) == 'ALEX WAS TIRED SO HE TOOK A NAP'
        assert set(assembly['word_num'].values) == {0, 1, 2, 3, 4, 5, 6, 7}
        assert len(set(assembly['sentence_id'].values)) == 52
        assert len(assembly['neuroid']) == 97
        assert len(np.unique(assembly['subject_UID'])) == 5

    def test_Blank2014(self):
        assembly = load_dataset('Blank2014.fROI')
        assert len(assembly['presentation']) == 1317
        assembly[assembly.where((assembly['story'] == 'Boar') & (assembly['sentence_num'] <= 10))]
        assert assembly.sel(story='Boar')['stimulus_sentence']
        assert 'Once upon a time' in assembly['stimuli'].values
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
        metric = load_metric('linear_pearsonr')
        score = metric(assembly1=assembly, assembly2=assembly)
        assert score.sel(aggregation='center') == approx(1, abs=.00001)

    def test_offset_source_target(self):
        source = self._make_assembly()
        target = source + 2  # offset all values
        metric = load_metric('linear_pearsonr')
        score = metric(assembly1=source, assembly2=target)
        assert score.sel(aggregation='center') == approx(1, abs=.00001)

    def test_mismatched_source_target(self):
        random_state = RandomState(1)
        source = (np.arange(30 * 25) + random_state.standard_normal(30 * 25)).reshape((30, 25))
        source = self._make_assembly(source)
        target = random_state.poisson(lam=1, size=30 * 25).reshape((30, 25))
        target = self._make_assembly(target)
        metric = load_metric('linear_pearsonr')
        score = metric(assembly1=source, assembly2=target)
        assert score.sel(aggregation='center') == approx(.02826294, abs=.00001)

    def test_weights_stored(self):
        assembly = self._make_assembly()
        metric = load_metric('linear_pearsonr', store_regression_weights=True)
        score = metric(assembly1=assembly, assembly2=assembly)
        assert score.attrs['regression_coef'] is not None
        assert score.attrs['regression_intercept'] is not None

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


class TestBenchmark:
    class DummyModel(ArtificialSubject):
        def __init__(self, neural_activity):
            self.neural_activity = neural_activity

        def digest_text(self, stimuli):
            np.testing.assert_array_equal(stimuli, self.neural_activity['stimulus'])
            return {'neural': self.neural_activity}

        def perform_neural_recording(self, recording_target: ArtificialSubject.RecordingTarget,
                                     recording_type: ArtificialSubject.RecordingType):
            assert recording_target == ArtificialSubject.RecordingTarget.language_system
            assert recording_type == ArtificialSubject.RecordingType.spikerate_exact

    def test_dummy_bad(self):
        benchmark = load_benchmark('Pereira2018-linear')
        neural_activity = RandomState(0).random(size=(30, 25))  # todo
        neural_activity = NeuroidAssembly(neural_activity,
                                          coords={'stimulus_id': ('presentation', np.arange(30)),
                                                  'stimulus_category': ('presentation', ['a', 'b', 'c'] * 10),
                                                  'neuroid_id': ('neuroid', np.arange(25)),
                                                  'region': ('neuroid', ['some_region'] * 25)},
                                          dims=['presentation', 'neuroid'])
        dummy_model = TestBenchmark.DummyModel(neural_activity=neural_activity)
        score = benchmark(dummy_model)
        assert score == approx(0.0098731 / .318567, abs=0.001)

    def test_exact(self):
        benchmark = load_benchmark('Pereira2018-linear')
        dummy_model = TestBenchmark.DummyModel(neural_activity=benchmark.data)
        score = benchmark(dummy_model)
        assert score == approx(1)

    def test_ceiling(self):
        benchmark = load_benchmark('Pereira2018-linear')
        ceiling = benchmark.ceiling
        assert ceiling == approx(.318567, abs=.0005)
