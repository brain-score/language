import argparse
from typing import List, Union, Dict

from brainscore_core import Score, Benchmark
from brainscore_core.submission import UserManager, RunScoringEndpoint, DomainPlugins
from brainscore_language import load_model, load_benchmark, score
from brainscore_language.submission import config


def process_github_submission(plugin_info: Dict[str, Union[List[str], str]]):
    """
    Triggered when changes are merged to the GitHub repository, if those changes affect benchmarks or models.
    Starts run to score models on benchmarks (`run_scoring`).
    """
    jenkins_base = "http://braintree.mit.edu:8080"
    jenkins_user = os.environ['JENKINS_USER']
    jenkins_token = os.environ['JENKINS_TOKEN']
    jenkins_trigger = os.environ['JENKINS_TRIGGER']
    jenkins_job = "dev_score_plugins"

    url = f'{jenkins_base}/job/{jenkins_job}/buildWithParameters?token={jenkins_trigger}'
    payload = {k: v for k, v in plugin_info.items() if plugin_info[k]}
    auth_basic = HTTPBasicAuth(username=jenkins_user, password=jenkins_token)
    r = requests.get(url, params=payload, auth=auth_basic)
    logger.debug(r)


class LanguagePlugins(DomainPlugins):
    def load_model(self, model_identifier: str):
        return load_model(model_identifier)

    def load_benchmark(self, benchmark_identifier: str) -> Benchmark:
        return load_benchmark(benchmark_identifier)

    def score(self, model_identifier: str, benchmark_identifier: str) -> Score:
        return score(model_identifier, benchmark_identifier)


language_plugins = LanguagePlugins()
run_scoring_endpoint = RunScoringEndpoint(language_plugins, db_secret=config.get_database_secret())


def get_user_email(uid: int) -> str:
    """ Convenience method for GitHub Actions to get a user's email if their web-submitted PR fails. """
    return get_email_from_uid(uid)


def create_user(domain: str, email: str) -> int:
    user_manager = UserManager(domain, email, db_secret=config.get_database_secret())
    new_user_id = user_manager()
    return new_user_id


def _get_ids(args_dict: Dict[str, Union[str, List]], key: str) -> Union[List, str, None]:
    return args_dict[key] if key in args_dict else None


def _clean_args(remove_keys: List[str], args_dict: Dict[str, Union[str, List]]) -> Dict[str, Union[str, List]]:
    return {k: v for k, v in args_dict.items() if k not in remove_keys}  # preserve other keys, e.g. `run_score`


def run_scoring(args_dict: Dict[str, Union[str, List]]):
    """ prepares `args_dict` as parameters for the `run_scoring_endpoint`. """
    new_models = _get_ids(args_dict, 'new_models')
    new_benchmarks = _get_ids(args_dict, 'new_benchmarks')

    if args_dict['specified_only']:
        assert len(new_models) > 0, "No models specified"
        assert len(new_benchmarks) > 0, "No benchmarks specified"
        args_dict['models'] = new_models
        args_dict['benchmarks'] = new_benchmarks
    else:
        if new_models and new_benchmarks:
            args_dict['models'] = RunScoringEndpoint.ALL_PUBLIC
            args_dict['benchmarks'] = RunScoringEndpoint.ALL_PUBLIC
        elif new_benchmarks:
            args_dict['models'] = RunScoringEndpoint.ALL_PUBLIC
            args_dict['benchmarks'] = new_benchmarks
        elif new_models:
            args_dict['models'] = new_models
            args_dict['benchmarks'] = RunScoringEndpoint.ALL_PUBLIC

    new_args = _clean_args(['new_benchmarks', 'new_models', 
                            'author_email', 'specified_only'], args_dict)
    new_args["domain"] = "language"

    run_scoring_endpoint(**new_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('jenkins_id', type=int,
                        help='The id of the current jenkins run')
    parser.add_argument('domain', type=str,
                        help='The submission domain (vision or language)')
    parser.add_argument('user_id', type=int, nargs='?', default=None,
                        help='ID of submitting user in the postgres DB')
    parser.add_argument('author_email', type=str, nargs='?', default=None,
                        help='email associated with PR author GitHub username')
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
    parser.add_argument('--specified_only', type=bool, nargs='?', default=False,
                        help='Only score the plugins specified by new_models and new_benchmarks')
    args, remaining_args = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    args_dict = vars(args)

    if not args_dict['user_id']:
        new_user_id = create_user(args_dict['domain'], args_dict['author_email'])
        args_dict['user_id'] = new_user_id
    
    run_scoring(new_args)
