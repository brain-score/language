from pathlib import Path
import pickle
import subprocess
from typing import Dict, Any, Type, Union

from brainio.assemblies import DataAssembly
from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Score, Metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.plugin_management.conda_score import CondaScore, get_score, save_score

data_registry: Dict[str, Type[Union[DataAssembly, Any]]] = {}
""" Pool of available data """

metric_registry: Dict[str, Type[Metric]] = {}
""" Pool of available metrics """

benchmark_registry: Dict[str, Type[Benchmark]] = {}
""" Pool of available benchmarks """

model_registry: Dict[str, Type[ArtificialSubject]] = {}
""" Pool of available models """


def locate_plugin(plugin_type: str, identifier: str) -> str:
    plugins_dir = Path(__file__).with_name(plugin_type)
    plugins = [d.name for d in plugins_dir.iterdir() if d.is_dir()]

    specified_plugin_dirname = None

    for plugin_dirname in plugins:
        plugin_dirpath = plugins_dir / plugin_dirname
        init_file = plugin_dirpath / "__init__.py"
        with open(init_file) as f:
            registry_name = plugin_type.strip('s') + '_registry'
            plugin_registrations = [line for line in f if f'{registry_name}[\'{identifier}\']'
                                        in line.replace('\"', '\'')]
            if len(plugin_registrations) > 0:
                specified_plugin_dirname = plugin_dirname

    return specified_plugin_dirname


def _install_requirements(plugin_type:str, plugin_dirname: str):
    plugins_dir = Path(__file__).with_name(plugin_type)
    requirements_file = plugins_dir / plugin_dirname / 'requirements.txt'
    if requirements_file.is_file():
        subprocess.run(f"pip install -r {requirements_file}", shell=True)


def import_plugin(plugin_type: str, identifier: str):
    plugin_dirname = locate_plugin(plugin_type, identifier)
    _install_requirements(plugin_type, plugin_dirname)
    __import__(f'brainscore_language.{plugin_type}.{plugin_dirname}')


def load_dataset(identifier: str) -> Union[DataAssembly, Any]:
    import_plugin('data', identifier)

    return data_registry[identifier]()


def load_metric(identifier: str, *args, **kwargs) -> Metric:
    import_plugin('metrics', identifier)

    return metric_registry[identifier](*args, **kwargs)


def load_benchmark(identifier: str) -> Benchmark:
    import_plugin('benchmarks', identifier)

    return benchmark_registry[identifier]()


def load_model(identifier: str) -> ArtificialSubject:
    import_plugin('models', identifier)

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
    if create_env: # create conda environment
        plugin_ids = {'model':model_identifier, 'benchmark':benchmark_identifier}
        CondaScore(plugin_ids)
        result = read_score()
    else:
        result = score(model_identifier, benchmark_identifier)

    return result
