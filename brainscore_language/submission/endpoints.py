import argparse

from brainscore_core import Score, Benchmark
from brainscore_core.submission import RunScoringEndpoint, DomainPlugins
from brainscore_language import load_model, load_benchmark, score
from brainscore_language.submission import config

from typing import List, Union


class LanguagePlugins(DomainPlugins):
    def load_model(self, model_identifier: str):
        return load_model(model_identifier)

    def load_benchmark(self, benchmark_identifier: str) -> Benchmark:
        return load_benchmark(benchmark_identifier)

    def score(self, model_identifier: str, benchmark_identifier: str) -> Score:
        return score(model_identifier, benchmark_identifier)

language_plugins = LanguagePlugins()
run_scoring_endpoint = RunScoringEndpoint(language_plugins, db_secret=config.get_database_secret())

def _not_empty(plugin_list):
    return any(s.strip() for s in plugin_list)

def run_scoring(args_dict):
    new_models = args_dict['new_models']
    new_benchmarks = args_dict['new_benchmarks']
    all_models = args_dict['all_models']
    all_benchmarks = args_dict['all_benchmarks']

    if _not_empty(new_models) and _not_empty(new_benchmarks):
        args_dict['models'] = all_models
        args_dict['benchmarks'] = all_benchmarks
    elif _not_empty(new_benchmarks):
        args_dict['models'] = all_models
        args_dict['benchmarks'] = new_benchmarks
    elif _not_empty(new_models):
        args_dict['models'] = new_models
        args_dict['benchmarks'] = all_benchmarks

    remove_keys = ['new_benchmarks', 'new_models', 'all_benchmarks', 'all_models']
    new_args = {k:v for k,v in args_dict.items() if k not in remove_keys}

    return run_scoring_endpoint(**new_args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('jenkins_id', type=int,
                        help='The id of the current jenkins run')
    parser.add_argument('--new_models', type=str, nargs='*', default=None,
                        help='The identifiers of newly submitted models to score on all benchmarks')
    parser.add_argument('--new_benchmarks', type=str, nargs='*', default=None,
                        help='The identifiers of newly submitted benchmarks on which to score all models')
    parser.add_argument('--all_models', type=str, nargs='*', default=None,
                        help='All registered models')
    parser.add_argument('--all_benchmarks', type=str, nargs='*', default=None,
                        help='All registered benchmarks')
    parser.add_argument('user_id', type=int, nargs='?', default=2,
                        help='ID of submitting user in the postgres DB')
    parser.add_argument('model_type', type=str, nargs='?', default='artificialsubject',
                        help='Type of model to score')
    parser.add_argument('public', type=bool, nargs='?', default=True,
                        help='Public (or private) submission?')
    parser.add_argument('competition', type=str, nargs='?', default=None,
                        help='Name of competition for which submission is being scored')
    args, remaining_args = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    run_scoring(vars(args))
