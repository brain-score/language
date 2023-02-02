import argparse
from typing import List, Union, Dict

from brainscore_core import Score, Benchmark
from brainscore_core.submission import RunScoringEndpoint, DomainPlugins
from brainscore_language import load_model, load_benchmark, score
from brainscore_language.submission import config


class LanguagePlugins(DomainPlugins):
    def load_model(self, model_identifier: str):
        return load_model(model_identifier)

    def load_benchmark(self, benchmark_identifier: str) -> Benchmark:
        return load_benchmark(benchmark_identifier)

    def score(self, model_identifier: str, benchmark_identifier: str) -> Score:
        return score(model_identifier, benchmark_identifier)


language_plugins = LanguagePlugins()
run_scoring_endpoint = RunScoringEndpoint(language_plugins, db_secret=config.get_database_secret())


def _get_ids(args_dict: Dict[str, Union[str, List]], key: str) -> Union[List, str, None]:
    return args_dict[key] if key in args_dict else None


def run_scoring(args_dict: Dict[str, Union[str, List]]):
    """ prepares `args_dict` as parameters for the `run_scoring_endpoint`. """
    new_models = _get_ids(args_dict, 'new_models')
    new_benchmarks = _get_ids(args_dict, 'new_benchmarks')

    print(f"new models: {new_models}")
    print(f"new benchmarks: {new_benchmarks}")

    if new_models and new_benchmarks:
        args_dict['models'] = RunScoringEndpoint.ALL_PUBLIC
        args_dict['benchmarks'] = RunScoringEndpoint.ALL_PUBLIC
    elif new_benchmarks:
        args_dict['models'] = RunScoringEndpoint.ALL_PUBLIC
        args_dict['benchmarks'] = new_benchmarks
    elif new_models:
        args_dict['models'] = new_models
        args_dict['benchmarks'] = RunScoringEndpoint.ALL_PUBLIC

    remove_keys = ['new_benchmarks', 'new_models']
    new_args = {k: v for k, v in args_dict.items() if k not in remove_keys}  # preserve other keys, e.g. `run_score`
    print(new_args)

    run_scoring_endpoint(**new_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('jenkins_id', type=int,
                        help='The id of the current jenkins run')
    parser.add_argument('user_id', type=int, nargs='?', default=2,
                        help='ID of submitting user in the postgres DB')
    parser.add_argument('model_type', type=str, nargs='?', default='artificialsubject',
                        help='Type of model to score')
    parser.add_argument('public', type=bool, nargs='?', default=True,
                        help='Public (or private) submission?')
    parser.add_argument('competition', type=str, nargs='?', default=None,
                        help='Name of competition for which submission is being scored')
    parser.add_argument('--new_models', type=str, nargs='*', default=None,
                        help='The identifiers of newly submitted models to score on all benchmarks')
    parser.add_argument('--new_benchmarks', type=str, nargs='*', default=None,
                        help='The identifiers of newly submitted benchmarks on which to score all models')
    args, remaining_args = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    run_scoring(vars(args))
