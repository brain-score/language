from typing import List, Union, Dict

from brainscore_core import Score, Benchmark
from brainscore_core.submission import RunScoringEndpoint, DomainPlugins
from brainscore_core.submission.endpoints import make_argparser, retrieve_models_and_benchmarks, resolve_models, resolve_benchmarks, get_user_id, \
    send_email_to_submitter as send_email_to_submitter_core
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


def run_scoring(args_dict: Dict[str, Union[str, List]]):
    model_ids, benchmark_ids = resolve_models_benchmarks(args_dict)
    
    for benchmark in benchmark_ids:
        for model in model_ids:
            run_scoring_endpoint(domain="language", jenkins_id=args_dict["jenkins_id"],
                                model_identifier=model, benchmark_identifier=benchmark,
                                user_id=args_dict["user_id"], model_type="artificialsubject",
                                public=args_dict["public"], competition=args_dict["competition"])

def resolve_models_benchmarks(args_dict: Dict[str, Union[str, List]]):
    benchmarks, models = retrieve_models_and_benchmarks(args_dict)

    model_ids = resolve_models(models)
    benchmark_ids = resolve_benchmarks(benchmarks)
    print("BS_NEW_MODELS=" + " ".join(model_ids))
    print("BS_NEW_BENCHMARKS=" + " ".join(benchmark_ids))
    return model_ids, benchmark_ids

def send_email_to_submitter(uid: int, domain: str, pr_number: str,
                            mail_username: str, mail_password: str):
    send_email_to_submitter_core(uid=uid, domain=domain, pr_number=pr_number,
                                 db_secret=config.get_database_secret(),
                                 mail_username=mail_username, mail_password=mail_password)


if __name__ == '__main__':
    parser = make_argparser()
    parser.add_argument('--fn', type=str, nargs='?', default='run_scoring',
                    choices=['run_scoring', 'retrieve_models_and_benchmarks'],
                    help='The endpoint method to run. `run_scoring` to score `new_models` on `new_benchmarks`, or `retrieve_models_and_benchmarks` to respond with a list of models and benchmarks to score.')

    args, remaining_args = parser.parse_known_args()
    args_dict = vars(args)

    if 'user_id' not in args_dict or args_dict['user_id'] is None:
        user_id = get_user_id(args_dict['author_email'], db_secret=config.get_database_secret())
        args_dict['user_id'] = user_id
    
    if args.fn == 'run_scoring':
        run_scoring(args_dict)
    elif args.fn == 'retrieve_models_and_benchmarks':
        retrieve_models_and_benchmarks(args_dict)
    else:
        raise ValueError(f'Invalid method: {args.fn}')
