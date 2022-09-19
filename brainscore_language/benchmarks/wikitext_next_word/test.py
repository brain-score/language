import numpy as np
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_language import load_benchmark
from brainscore_language.artificial_subject import ArtificialSubject


class TestBenchmark:
    class DummyModel(ArtificialSubject):
        def digest_text(self, stimuli):
            return {'behavior': BehavioralAssembly(
                ['the' for passage in stimuli],
                coords={'stimulus': ('presentation', stimuli), 'stimulus_id': ('presentation', np.arange(len(stimuli)))},
                dims=['presentation'])}

        def perform_behavioral_task(self, task: ArtificialSubject.Task):
            if task != ArtificialSubject.Task.next_word:
                raise NotImplementedError()

    def test_dummy_the(self):
        benchmark = load_benchmark('Wikitext-accuracy')
        dummy_model = TestBenchmark.DummyModel()
        score = benchmark(dummy_model)
        assert score == approx(0.05945, abs=0.001)
