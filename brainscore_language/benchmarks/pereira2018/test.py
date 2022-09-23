import copy

import numpy as np
import pytest
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
            assert len(stimuli) == len(self.neural_activity['presentation'])
            self.neural_activity['stimulus'] = 'presentation', stimuli  # copy over
            return {'neural': self.neural_activity}

        def perform_neural_recording(self, recording_target: ArtificialSubject.RecordingTarget,
                                     recording_type: ArtificialSubject.RecordingType):
            assert recording_target == ArtificialSubject.RecordingTarget.language_system
            assert recording_type == ArtificialSubject.RecordingType.fMRI

    @pytest.mark.parametrize('experiment, expected_score', [
        (243, approx(0.0017534 / .35378928, abs=0.001)),
        (384, approx(0.01215216 / .36343748, abs=0.001)),
    ])
    def test_dummy_bad(self, experiment, expected_score):
        benchmark = load_benchmark(f'Pereira2018.{experiment}sentences-linear')
        neural_activity = RandomState(0).random(size=(experiment, 25))  # presentation x neuroid
        neural_activity = NeuroidAssembly(neural_activity,
                                          coords={'stimulus_seq': ('presentation', np.arange(experiment)),
                                                  'stimulus_num': ('presentation', np.arange(experiment)),
                                                  'neuroid_id': ('neuroid', np.arange(25)),
                                                  'region': ('neuroid', ['some_region'] * 25)},
                                          dims=['presentation', 'neuroid'])
        dummy_model = TestBenchmark.DummyModel(neural_activity=neural_activity)
        score = benchmark(dummy_model)
        assert score == expected_score

    @pytest.mark.parametrize('experiment', [
        243,
        384,
    ])
    def test_exact(self, experiment):
        benchmark = load_benchmark(f'Pereira2018.{experiment}sentences-linear')
        exact_data = copy.deepcopy(benchmark.data).reset_index('presentation')
        del exact_data['stimulus_id'], exact_data['stimulus']
        exact_data = exact_data.set_index(presentation=list(exact_data['presentation'].coords))
        dummy_model = TestBenchmark.DummyModel(neural_activity=exact_data)
        score = benchmark(dummy_model)
        assert score == approx(1)

    @pytest.mark.parametrize('experiment, expected_ceiling', [
        (243, .35378928),
        (384, .36343748),
    ])
    def test_ceiling(self, experiment, expected_ceiling):
        benchmark = load_benchmark(f'Pereira2018.{experiment}sentences-linear')
        ceiling = benchmark.ceiling
        assert ceiling == approx(expected_ceiling, abs=.0005)

    @pytest.mark.parametrize('experiment', [243, 384])
    def test_ceiling_raw(self, experiment):
        benchmark = load_benchmark(f'Pereira2018.{experiment}sentences-linear')
        ceiling = benchmark.ceiling
        assert hasattr(ceiling, 'raw')
        assert set(ceiling.raw.dims) == {'neuroid'}
        assert ceiling.raw.median() == ceiling
        assert hasattr(ceiling.raw, 'raw')
        assert set(ceiling.raw.raw.dims) == {'sub_subject', 'num_subjects', 'split', 'neuroid'}
