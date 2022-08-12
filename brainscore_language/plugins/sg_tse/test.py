from pytest import approx
from brainscore_language import load_benchmark, load_metric
from brainscore_language.models.huggingface import HuggingfaceSubject
from brainscore_language.plugins.sg_tse import _load_suite

class TestData:
    def test_firstline(self):
        suite = _load_suite('test_suite.json')
        data = list(suite.iter_sentences())
        assert data[0] == 'The dog sleeps on the mat'

    def test_length(self):
        suite = _load_suite('test_suite.json')
        data = list(suite.iter_sentences())
        assert len(data[0]) == 25

class TestMetric:
    def test_all_correct(self):
        predictions = ["True", "True", "True"]
        targets = ["True", "True", "True"]
        metric = load_metric('accuracy')
        score = metric(predictions, targets)
        assert score == 1

    def test_half_correct(self):
        predictions = ["True", "False"]
        targets = ["True", "True"]
        metric = load_metric('accuracy')
        score = metric(predictions, targets)
        assert score == 0.5

    def test_none_correct(self):
        predictions = ["False", "False", "False"]
        targets = ["True", "True", "True"]
        metric = load_metric('accuracy')
        score = metric(predictions, targets)
        assert score == 0

class TestBenchmark:
    class DummyModel(HuggingfaceSubject):
            def perform_task(self, task: HuggingfaceSubject.Task):
                if task != HuggingfaceSubject.Task.next_word:
                    raise NotImplementedError()

    def test_dummy_the(self):
        benchmark = load_benchmark('SG-TSE')
        dummy_model = TestBenchmark.DummyModel(model_id='gpt2', region_layer_mapping={
            HuggingfaceSubject.RecordingTarget.language_system: 'transformer.h.0.ln_1'})
        score = benchmark(dummy_model)
        assert score == approx(0.3333, abs=0.001)


