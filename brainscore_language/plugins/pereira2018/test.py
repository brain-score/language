import numpy as np
from numpy.random import RandomState
from pytest import approx

from brainio.assemblies import NeuroidAssembly
from brainscore_language import load_dataset, ArtificialSubject, load_benchmark


class TestData:
    def test_language(self):
        assembly = load_dataset('Pereira2018.language')
        assert set(assembly['experiment'].values) == {'243sentences', '384sentences'}
        assert len(assembly['presentation']) == 243 + 384
        assert len(set(assembly['stimulus'].values)) == 243 + 384
        assert 'The concert pianist went blind in adulthood.' in assembly['stimulus'].values
        assert len(set(assembly['subject'].values)) == 10
        assert len(set(assembly['neuroid_id'].values)) == 13553
        assert np.nansum(assembly.values) == approx(1935595.263162177)

    def test_auditory(self):
        assembly = load_dataset('Pereira2018.auditory')
        assert set(assembly['experiment'].values) == {'243sentences', '384sentences'}
        assert len(assembly['presentation']) == 243 + 384
        assert len(set(assembly['stimulus'].values)) == 243 + 384
        assert 'The concert pianist went blind in adulthood.' in assembly['stimulus'].values
        assert len(set(assembly['subject'].values)) == 10
        assert len(set(assembly['neuroid_id'].values)) == 5692
        assert np.nansum(assembly.values) == approx(-257124.1144940494)


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
