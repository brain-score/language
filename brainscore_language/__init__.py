from typing import Dict, Any, Type

from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Score, Metric
from brainscore_language.artificial_subject import ArtificialSubject

datasets: Dict[str, Type[Any]] = {}
""" Pool of available data """

metrics: Dict[str, Type[Metric]] = {}
""" Pool of available metrics """

benchmarks: Dict[str, Type[Benchmark]] = {}
""" Pool of available benchmarks """

models: Dict[str, ArtificialSubject] = {}
""" Pool of available models """


def load_dataset(identifier: str):
    # imports to load plugins until plugin system is implemented
    from brainscore_language.plugins.wikitext_next_word_prediction import data
    from brainscore_language.plugins.schrimpf2021 import data

    return datasets[identifier]()


def load_metric(identifier: str) -> Metric:
    # imports to load plugins until plugin system is implemented
    from brainscore_language.plugins.wikitext_next_word_prediction import metric
    from brainscore_language.plugins.schrimpf2021 import metric

    return metrics[identifier]()


def load_benchmark(identifier: str) -> Benchmark:
    # imports to load plugins until plugin system is implemented
    from brainscore_language.plugins.wikitext_next_word_prediction import benchmark
    from brainscore_language.plugins.schrimpf2021 import benchmark

    return benchmarks[identifier]()


def load_model(identifier: str) -> ArtificialSubject:
    return models[identifier]


def score(model_identifier: str, benchmark_identifier: str) -> Score:
    """
    Score the model referenced by the `model_identifier` on the benchmark referenced by the `benchmark_identifier`.
    The model needs to implement the :class:`~brainscore_language.artificial_subject.ArtificialSubject` interface
    so that the benchmark can interact with it.
    The benchmark will be looked up from the :data:`~brainscore_language.benchmarks` and evaluates the model on how
    brain-like it is under that benchmark's experimental paradigm, primate measurements, comparison metric, and ceiling
    This results in a quantitative
    `Score <https://brain-score-core.readthedocs.io/en/latest/modules/metrics.html#brainscore_core.metrics.Score>`_
    ranging from 0 (least brain-like) to 1 (most brain-like under this benchmark).

    :param model_identifier: the identifier for the model
    :param benchmark_identifier: the identifier for the benchmark to test the model against
    :return: a Score of how brain-like the candidate model is under this benchmark. The score is normalized by
        this benchmark's ceiling such that 1 means the model matches the data to ceiling level.
    """
    model: ArtificialSubject = load_model(model_identifier)
    benchmark: Benchmark = load_benchmark(benchmark_identifier)
    score: Score = benchmark(model)
    return score
