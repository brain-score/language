import argparse
from typing import List, Union, Dict

from brainscore_core import Score, Benchmark
from brainscore_core.submission import UserManager, RunScoringEndpoint, DomainPlugins
from brainscore_language import load_model, load_benchmark, score
from brainscore_language.submission import config


def call_jenkins(plugin_info: Dict[str, Union[List[str], str]]):
    """
    Triggered when changes are merged to the GitHub repository, if those changes affect benchmarks or models.
    Starts run to score models on benchmarks (`run_scoring`).
    """
    jenkins_base = "http://braintree.mit.edu:8080"
    jenkins_user = os.environ['JENKINS_USER']
    jenkins_token = os.environ['JENKINS_TOKEN']
    jenkins_trigger = os.environ['JENKINS_TRIGGER']
    jenkins_job = "score_plugins"

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


def send_email_to_submitter(uid: int, domain: str, pr_number: str, 
                            mail_username: str, mail_password:str ) -> str:
    """ Send submitter an email if their web-submitted PR fails. """
    subject = "Brain-Score submission failed"
    body = f"Your Brain-Score submission did not pass checks. Please review the test results and update the PR at https://github.com/brain-score/{domain}/pull/{pr_number} or send in an updated submission via the website."
    return send_user_email(uid, body, mail_username, mail_password)


def get_user_id(email: str) -> int:
    user_manager = UserManager(db_secret=config.get_database_secret())
    user_id = user_manager.get_uid(email)
    return user_id


def _get_ids(args_dict: Dict[str, Union[str, List]], key: str) -> Union[List, str, None]:
    return args_dict[key] if key in args_dict else None


def run_scoring(args_dict: Dict[str, Union[str, List]]):
    """ prepares parameters for the `run_scoring_endpoint`. """
    new_models = _get_ids(args_dict, 'new_models')
    new_benchmarks = _get_ids(args_dict, 'new_benchmarks')

    if args_dict['specified_only']:
        assert len(new_models) > 0, "No models specified"
        assert len(new_benchmarks) > 0, "No benchmarks specified"
        models = new_models
        benchmarks = new_benchmarks
    else:
        if new_models and new_benchmarks:
            models = RunScoringEndpoint.ALL_PUBLIC
            benchmarks = RunScoringEndpoint.ALL_PUBLIC
        elif new_benchmarks:
            models = RunScoringEndpoint.ALL_PUBLIC
            benchmarks = new_benchmarks
        elif new_models:
            models = new_models
            benchmarks = RunScoringEndpoint.ALL_PUBLIC

    run_scoring_endpoint(domain="language", jenkins_id=args_dict["jenkins_id"], 
        models=models, benchmarks=benchmarks, user_id=args_dict["user_id"], 
        model_type="artificialsubject", public=args_dict["public"], 
        competition=args_dict["competition"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('jenkins_id', type=int,
                        help='The id of the current jenkins run')
    parser.add_argument('public', type=bool, nargs='?', default=True,
                        help='Public (or private) submission?')
    parser.add_argument('--competition', type=str, nargs='?', default=None,
                        help='Name of competition for which submission is being scored')
    parser.add_argument('--user_id', type=int, nargs='?', default=None,
                        help='ID of submitting user in the postgres DB')
    parser.add_argument('--author_email', type=str, nargs='?', default=None,
                        help='email associated with PR author GitHub username')
    parser.add_argument('--specified_only', type=bool, nargs='?', default=False,
                        help='Only score the plugins specified by new_models and new_benchmarks')
    parser.add_argument('--new_models', type=str, nargs='*', default=None,
                        help='The identifiers of newly submitted models to score on all benchmarks')
    parser.add_argument('--new_benchmarks', type=str, nargs='*', default=None,
                        help='The identifiers of newly submitted benchmarks on which to score all models')
    args, remaining_args = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    args_dict = vars(args)

    if 'user_id' not in args_dict or args_dict['user_id'] == None:
        user_id = get_user_id(args_dict['author_email'])
        args_dict['user_id'] = new_user_id
    
    run_scoring(args_dict)
