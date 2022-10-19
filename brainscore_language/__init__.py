from pathlib import Path
import pickle
import subprocess
from typing import Dict, Any, Type, Union

from brainio.assemblies import DataAssembly
from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Score, Metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.plugin_management.import_plugin import ImportPlugin
from brainscore_language.plugin_management.conda_score import CondaScore, save_score, read_score

data_registry: Dict[str, Type[Union[DataAssembly, Any]]] = {}
""" Pool of available data """

metric_registry: Dict[str, Type[Metric]] = {}
""" Pool of available metrics """

benchmark_registry: Dict[str, Type[Benchmark]] = {}
""" Pool of available benchmarks """

model_registry: Dict[str, Type[ArtificialSubject]] = {}
""" Pool of available models """


def load_dataset(identifier: str) -> Union[DataAssembly, Any]:
    ImportPlugin('data', identifier)

    return data_registry[identifier]()


def load_metric(identifier: str, *args, **kwargs) -> Metric:
    ImportPlugin('metrics', identifier)

    return metric_registry[identifier](*args, **kwargs)


def load_benchmark(identifier: str) -> Benchmark:
    ImportPlugin('benchmarks', identifier)

    return benchmark_registry[identifier]()


def load_model(identifier: str) -> ArtificialSubject:
    ImportPlugin('models', identifier)

    model = model_registry[identifier]()
    model.identifier = identifier

    return model


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
    score.attrs['model_identifier'] = model_identifier
    score.attrs['benchmark_identifier'] = benchmark_identifier
    save_score(score)
    
    return score


def get_score(model_identifier: str, benchmark_identifier: str, create_env=False) -> Score:
    """ if create_env, runs score() in a conda environment """
    if create_env:
        CondaScore(model_identifier, benchmark_identifier)
        result = read_score()
    else:
        result = score(model_identifier, benchmark_identifier)

    return result
