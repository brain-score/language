import pytest

from brainscore_language.plugin_management.conda_score import CondaScore

@pytest.mark.memory_intense
def test_score_in_env():
	model_identifier = 'distilgpt2'
	benchmark_identifier = 'Futrell2018-pearsonr'
	plugin_ids = {'model': model_identifier, 'benchmark':benchmark_identifier}

	conda_score = CondaScore(plugin_ids)
	result = conda_score().results
	print(result)
	# assert plugin_test_runner.results[plugin_test_runner.plugin_name] == 0
