import fire
import os

from brainscore_language import score as _score_function


def score(model_identifier: str, benchmark_identifier: str, install_dependencies=None):

    os.environ['BSL_INSTALL_DEPENDENCIES'] = install_dependencies or 'yes'
    
    result = _score_function(model_identifier, benchmark_identifier)
    print(result)  # print instead of return because fire has issues with xarray objects


if __name__ == '__main__':
    fire.Fire()