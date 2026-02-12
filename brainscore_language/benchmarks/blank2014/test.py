import copy
import numpy as np
from numpy.random import RandomState
from pytest import approx
from typing import Callable, Union, List

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_language import ArtificialSubject, load_benchmark


class TestBenchmark:
    class DummyModel(ArtificialSubject):
        def __init__(self, activity_for_text: Callable[[Union[str, List[str]]], NeuroidAssembly]):
            self.activity_for_text = activity_for_text

        def digest_text(self, stimuli):
            neural_activity = self.activity_for_text(stimuli)
            return {'neural': neural_activity}

        def start_neural_recording(self, recording_target: ArtificialSubject.RecordingTarget,
                                   recording_type: ArtificialSubject.RecordingType):
            assert recording_target == ArtificialSubject.RecordingTarget.language_system
            assert recording_type == ArtificialSubject.RecordingType.fMRI

    def test_dummy_bad(self):
        random_state = RandomState(0)

        def activity_for_text(stimuli: Union[str, List[str]]) -> NeuroidAssembly:
            num_stimuli = len(stimuli)
            num_neuroids = 25
            neural_activity = random_state.random(size=(num_stimuli, num_neuroids))  # presentation x neuroid
            neural_activity = NeuroidAssembly(neural_activity,
                                              coords={'stimulus_seq': ('presentation', np.arange(num_stimuli)),
                                                      'stimulus_num': ('presentation', np.arange(num_stimuli)),
                                                      'neuroid_id': ('neuroid', np.arange(num_neuroids)),
                                                      'region': ('neuroid', ['some_region'] * num_neuroids),
                                                      'layer': ('neuroid', ['test_layer'] * num_neuroids)},
                                              dims=['presentation', 'neuroid'])
            neural_activity['stimulus'] = 'presentation', stimuli  # copy over
            return neural_activity

        benchmark = load_benchmark('Blank2014-linear')
        dummy_model = TestBenchmark.DummyModel(activity_for_text=activity_for_text)
        score = benchmark(dummy_model)
        assert score == 0

    def test_exact(self):
        benchmark = load_benchmark('Blank2014-linear')
        exact_data = copy.deepcopy(benchmark.data)

        def activity_for_text(stimuli: Union[str, List[str]]) -> NeuroidAssembly:
            passage_activity = exact_data[{'presentation': [
                list(exact_data['stimulus'].values).index(stimulus) for stimulus in stimuli]}]
            # remove stimulus_id and stimulus coordinates to not trip up benchmark
            passage_activity = passage_activity.reset_index('presentation')
            del passage_activity['stimulus_id']
            passage_activity['layer'] = 'neuroid', ['test_layer'] * passage_activity.sizes['neuroid']
            passage_activity = NeuroidAssembly(passage_activity)  # index
            return passage_activity

        dummy_model = TestBenchmark.DummyModel(activity_for_text=activity_for_text)
        score = benchmark(dummy_model)
        assert score == approx(1)

    def test_ceiling(self):
        benchmark = load_benchmark(f'Blank2014-linear')
        ceiling = benchmark.ceiling
        assert ceiling == approx(.21026591, abs=.0005)

    def test_ceiling_raw(self):
        benchmark = load_benchmark(f'Blank2014-linear')
        ceiling = benchmark.ceiling
        assert hasattr(ceiling, 'raw')
        assert set(ceiling.raw.dims) == {'neuroid'}
        assert ceiling.raw.median() == ceiling
        assert hasattr(ceiling.raw, 'raw')
        assert set(ceiling.raw.raw.dims) == {'sub_subject_id', 'num_subjects', 'split', 'neuroid'}
