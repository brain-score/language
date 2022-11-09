import argparse

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

def run_scoring(args):
    return run_scoring_endpoint(**vars(args))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('jenkins_id', type=int,
                        help='The id of the current jenkins run')
    parser.add_argument('--model_identifier', type=str, default=None,
                        help='The identifier of the model to benchmark')
    parser.add_argument('--benchmark_identifier', type=str, default=None,
                        help='The identifier of the benchmark on which to score the model')
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
    run_scoring(args)
