import numpy as np
from numpy.random import RandomState
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_language import load_dataset, load_benchmark, load_metric
from brainscore_language.artificial_subject import ArtificialSubject


class TestData:
    def test_shape(self):
        assembly = load_dataset('Futrell2018')

        assert len(assembly['word']) == 10256
        assert len(set(assembly['stimulus_id'].values)) == len(assembly['presentation'])
        assert len(set(assembly['story_id'].values)) == 10
        assert len(set(assembly['sentence_id'].values)) == 481
        assert len(set(assembly['subject_id'].values)) == 180

        mean_assembly = assembly.mean('subject')
        assert not np.isnan(mean_assembly).any()


class TestMetric:
    def test_identical(self):
        a1 = a2 = [1, 2, 3, 4, 5]
        metric = load_metric('pearsonr')
        score = metric(a1, a2)
        assert score == 1

    def test_negative_correlation_is_1(self):
        a1 = np.array([1, 2, 3, 4, 5])
        a2 = -a1
        metric = load_metric('pearsonr')
        score = metric(a1, a2)
        assert score == 1

    def test_weak_correlation(self):
        a1 = [1, 2, 3, 4, 5]
        a2 = [3, 1, 6, 1, 2]
        metric = load_metric('pearsonr')
        score = metric(a1, a2)
        assert score == approx(.152, abs=.005)


class TestBenchmark:
    class DummyModel(ArtificialSubject):
        def __init__(self, reading_times):
            self.reading_times = reading_times

        def digest_text(self, stimuli):
            return {'behavior': BehavioralAssembly(
                self.reading_times,
                coords={'context': ('presentation', stimuli), 'stimulus_id': ('presentation', np.arange(len(stimuli)))},
                dims=['presentation'])}

        def perform_behavioral_task(self, task: ArtificialSubject.Task):
            if task != ArtificialSubject.Task.reading_times:
                raise NotImplementedError()

    def test_dummy_bad(self):
        benchmark = load_benchmark('Futrell2018-pearsonr')
        reading_times = RandomState(0).random(10256)
        dummy_model = TestBenchmark.DummyModel(reading_times=reading_times)
        score = benchmark(dummy_model)
        assert score == approx(0.0098731 / .858, abs=0.001)

    def test_exact(self):
        benchmark = load_benchmark('Futrell2018-pearsonr')
        dummy_model = TestBenchmark.DummyModel(reading_times=benchmark.data.mean('subject').values)
        score = benchmark(dummy_model)
        assert score == approx(1)

    def test_ceiling(self):
        benchmark = load_benchmark('Futrell2018-pearsonr')
        ceiling = benchmark.ceiling
        assert ceiling == approx(.858, abs=.0005)
