import numpy as np
from brainio.assemblies import BehavioralAssembly
from pytest import approx

from brainscore_language import load_dataset, load_benchmark, load_metric
from brainscore_language.artificial_subject import ArtificialSubject


class TestData:
    def test_firstline(self):
        data = load_dataset('wikitext-2/test')
        assert data[1] == ' = Robert Boulter = \n'

    def test_length(self):
        data = load_dataset('wikitext-2/test')
        assert len(data) == 4358


class TestMetric:
    def test_all_correct(self):
        predictions = targets = ["the", "quick", "brown", "fox", "jumped", "over"]
        metric = load_metric('accuracy')
        score = metric(predictions, targets)
        assert score == 1

    def test_half_correct(self):
        targets = ["the", "quick", "brown", "fox", "jumped", "over"]
        predictions = ["the", "slow", "brown", "cat", "jumped", "under"]
        metric = load_metric('accuracy')
        score = metric(predictions, targets)
        assert score == 0.5

    def test_none_correct(self):
        targets = ["the", "quick", "brown", "fox", "jumped", "over"]
        predictions = ["a", "slow", "white", "cat", "crawled", "under"]
        metric = load_metric('accuracy')
        score = metric(predictions, targets)
        assert score == 0


class TestBenchmark:
    class DummyModel(ArtificialSubject):
        def digest_text(self, stimuli):
            return {'behavior': BehavioralAssembly(
                ['the' for passage in stimuli],
                coords={'context': ('presentation', stimuli), 'stimulus_id': ('presentation', np.arange(len(stimuli)))},
                dims=['presentation'])}

        def perform_behavioral_task(self, task: ArtificialSubject.Task):
            if task != ArtificialSubject.Task.next_word:
                raise NotImplementedError()

    def test_dummy_the(self):
        benchmark = load_benchmark('Wikitext-accuracy')
        dummy_model = TestBenchmark.DummyModel()
        score = benchmark(dummy_model)
        assert score == approx(0.05945, abs=0.001)
