import os

from brainscore_language import benchmark_registry, metric_registry
from brainscore_language.plugin_management.import_plugin import ImportPlugin


def test_import_plugin():
	os.environ['BSL_DEPENDENCY_INSTALL'] = os.getenv('BSL_DEPENDENCY_INSTALL', 'newenv')
	assert len(benchmark_registry) == 0
	ImportPlugin('benchmarks', 'Futrell2018-pearsonr')
	assert 'Futrell2018-pearsonr' in benchmark_registry.keys()

def test_import_plugin_with_requirements():
	os.environ['BSL_DEPENDENCY_INSTALL'] = os.getenv('BSL_DEPENDENCY_INSTALL', 'newenv')
	assert len(metric_registry) == 0
	ImportPlugin('metrics', 'linear_pearsonr')
	assert 'linear_pearsonr' in metric_registry.keys()
