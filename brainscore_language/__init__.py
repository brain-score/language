from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Score
from brainscore_language.artificial_subject import ArtificialSubject

datasets = {}
metrics = {}
benchmarks = {}
models = {}


def load_dataset(identifier):
    # imports to load plugins until plugin system is implemented
    from brainscore_language.plugins.sg_targeted_syntactic_evaluation import data

    return datasets[identifier]()


def load_metric(identifier):
    # imports to load plugins until plugin system is implemented
    from brainscore_language.plugins.sg_targeted_syntactic_evaluation import metric

    return metrics[identifier]()


def load_benchmark(identifier):
    # imports to load plugins until plugin system is implemented
    from brainscore_language.plugins.sg_targeted_syntactic_evaluation import benchmark

    return benchmarks[identifier]()


def load_model(identifier):
    return models[identifier]


def score(model_identifier, benchmark_identifier) -> Score:
    model: ArtificialSubject = load_model(model_identifier)
    benchmark: Benchmark = load_benchmark(benchmark_identifier)
    score: Score = benchmark(model)
    return score
