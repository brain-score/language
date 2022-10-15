import fire

from brainscore_language import score as _score_function
from brainscore_language.plugin_management.conda_score import CondaScore


def score(model_identifier: str, benchmark_identifier: str, create_env=False):
    if create_env:
        # create conda environment from which score() is called
        plugin_ids = {'model':model_identifier, 'benchmark':benchmark_identifier}
        conda_score = CondaScore(plugin_ids)
        result = conda_score().results
    else:
        result = _score_function(model_identifier, benchmark_identifier)
    
    print(result)  # print instead of return because fire has issues with xarray objects


if __name__ == '__main__':
    fire.Fire()
