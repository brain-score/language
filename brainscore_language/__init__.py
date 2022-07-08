from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Score, Metric
from brainscore_language.artificial_subject import ArtificialSubject

datasets = {}
metrics = {}
benchmarks = {}
models = {}


def load_dataset(identifier):
    # imports to load plugins until plugin system is implemented
    from brainscore_language.plugins.wikitext_next_word_prediction import data

    return datasets[identifier]()


def load_metric(identifier) -> Metric:
    # imports to load plugins until plugin system is implemented
    from brainscore_language.plugins.wikitext_next_word_prediction import metric

    return metrics[identifier]()


def load_benchmark(identifier) -> Benchmark:
    # imports to load plugins until plugin system is implemented
    from brainscore_language.plugins.wikitext_next_word_prediction import benchmark

    return benchmarks[identifier]()


def load_model(identifier) -> ArtificialSubject:
    return models[identifier]


def score(model_identifier, benchmark_identifier) -> Score:
    model: ArtificialSubject = load_model(model_identifier)
    benchmark: Benchmark = load_benchmark(benchmark_identifier)
    score: Score = benchmark(model)
    return score
