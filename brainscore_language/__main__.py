from pathlib import Path

import fire

from brainscore_language import score as _score_function


def score(model_identifier: str, benchmark_identifier: str, conda_active: bool=False):
    result = _score_function(model_identifier, benchmark_identifier, conda_active)
    print(result)  # print instead of return because fire has issues with xarray objects


if __name__ == '__main__':
    fire.Fire()
