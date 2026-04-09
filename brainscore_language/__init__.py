from functools import partial
from typing import Dict, Any, Union, Callable

from brainscore_core.supported_data_standards.brainio.assemblies import DataAssembly
from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Score, Metric
from brainscore_core.plugin_management.conda_score import wrap_score
from brainscore_core.plugin_management.import_plugin import import_plugin
from brainscore_language.artificial_subject import ArtificialSubject

data_registry: Dict[str, Callable[[], Union[DataAssembly, Any]]] = {}
""" Pool of available data """

metric_registry: Dict[str, Callable[[], Metric]] = {}
""" Pool of available metrics """

benchmark_registry: Dict[str, Callable[[], Benchmark]] = {}
""" Pool of available benchmarks """

model_registry: Dict[str, Callable[[], ArtificialSubject]] = {}
""" Pool of available models """


def load_dataset(identifier: str) -> Union[DataAssembly, Any]:
    import_plugin('brainscore_language', 'data', identifier)

    return data_registry[identifier]()


def load_metric(identifier: str, *args, **kwargs) -> Metric:
    import_plugin('brainscore_language', 'metrics', identifier)

    return metric_registry[identifier](*args, **kwargs)


def load_benchmark(identifier: str) -> Benchmark:
    import_plugin('brainscore_language', 'benchmarks', identifier)

    return benchmark_registry[identifier]()


def load_model(identifier: str) -> 'UnifiedModel':
    from brainscore_core.model_interface import UnifiedModel
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter

    import_plugin('brainscore_language', 'models', identifier)

    model = model_registry[identifier]()

    if not isinstance(model, UnifiedModel):
        model.identifier = identifier
        model = LanguageModelAdapter(model)
    return model

def _run_score(model_identifier: str, benchmark_identifier: str,
               check_mem: bool = True) -> Score:
    """
    Score the model referenced by the `model_identifier` on the benchmark referenced by the `benchmark_identifier`.
    """
    from brainscore_core.compatibility import check_compatibility
    from brainscore_core.memory import check_memory

    model: ArtificialSubject = load_model(model_identifier)
    benchmark: Benchmark = load_benchmark(benchmark_identifier)

    # Pre-flight checks
    check_compatibility(model, benchmark)
    if check_mem:
        check_memory(model, benchmark)

    score: Score = benchmark(model)
    score.attrs['model_identifier'] = model_identifier
    score.attrs['benchmark_identifier'] = benchmark_identifier
    return score


def score(model_identifier: str, benchmark_identifier: str,
          conda_active: bool = False, check_mem: bool = True) -> Score:
    """
    Score the model referenced by the `model_identifier` on the benchmark referenced by the `benchmark_identifier`.
    The model needs to implement the :class:`~brainscore_language.artificial_subject.ArtificialSubject` interface
    so that the benchmark can interact with it.
    The benchmark will be looked up from the :data:`~brainscore_language.benchmarks` and evaluates the model
    (looked up from :data:`~brainscore_language.models`) on how brain-like it is under that benchmark's
    experimental paradigm, primate measurements, comparison metric, and ceiling.
    This results in a quantitative
    `Score <https://brain-score-core.readthedocs.io/en/latest/modules/metrics.html#brainscore_core.metrics.Score>`_
    ranging from 0 (least brain-like) to 1 (most brain-like under this benchmark).
    
    :param model_identifier: the identifier for the model
    :param benchmark_identifier: the identifier for the benchmark to test the model against
    :return: a Score of how brain-like the candidate model is under this benchmark. The score is normalized by
        this benchmark's ceiling such that 1 means the model matches the data to ceiling level.
    """
    score_fn = partial(_run_score, check_mem=check_mem)
    return wrap_score(__file__,
                      model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                      score_function=score_fn, conda_active=conda_active)
