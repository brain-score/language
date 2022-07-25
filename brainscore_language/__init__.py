from typing import Dict, Any, Type

from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Score, Metric
from brainscore_language.artificial_subject import ArtificialSubject

datasets: Dict[str, Type[Any]] = {}
metrics: Dict[str, Type[Metric]] = {}
benchmarks: Dict[str, Type[Benchmark]] = {}
models: Dict[str, ArtificialSubject] = {}


def load_dataset(identifier: str):
    # imports to load plugins until plugin system is implemented
    from brainscore_language.plugins.wikitext_next_word_prediction import data

    return datasets[identifier]()


def load_metric(identifier: str) -> Metric:
    # imports to load plugins until plugin system is implemented
    from brainscore_language.plugins.wikitext_next_word_prediction import metric

    return metrics[identifier]()


def load_benchmark(identifier: str) -> Benchmark:
    # imports to load plugins until plugin system is implemented
    from brainscore_language.plugins.wikitext_next_word_prediction import benchmark

    return benchmarks[identifier]()


def load_model(identifier: str) -> ArtificialSubject:
    return models[identifier]


def score(model_identifier: str, benchmark_identifier: str) -> Score:
    model: ArtificialSubject = load_model(model_identifier)
    benchmark: Benchmark = load_benchmark(benchmark_identifier)
    score: Score = benchmark(model)
    return score
