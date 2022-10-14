import fire

from brainscore_language import score as _score_function
from brainscore_language.plugin_management.environment_manager import EnvironmentManager


def score(model_identifier: str, benchmark_identifier: str):
    if True:
        identifiers = {'model':model_identifier, 'benchmark':benchmark_identifier}
        environment_manager = EnvironmentManager('score', identifiers=identifiers)

        result = environment_manager().results
    else:
        result = _score_function(model_identifier, benchmark_identifier)
    print(result)  # print instead of return because fire has issues with xarray objects


if __name__ == '__main__':
    fire.Fire()
