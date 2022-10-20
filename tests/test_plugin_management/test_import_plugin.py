import os

from brainscore_language import benchmark_registry, metric_registry
from brainscore_language.plugin_management.import_plugin import import_plugin


def test_import_plugin():
    os.environ['BSL_DEPENDENCY_INSTALL'] = os.getenv('BSL_DEPENDENCY_INSTALL', 'newenv')
    assert 'Wikitext-accuracy' not in benchmark_registry.keys()
    import_plugin('benchmarks', 'Wikitext-accuracy')
    assert 'Wikitext-accuracy' in benchmark_registry.keys()


def test_import_plugin_with_requirements():
    os.environ['BSL_DEPENDENCY_INSTALL'] = os.getenv('BSL_DEPENDENCY_INSTALL', 'newenv')
    assert 'pearsonr' not in benchmark_registry.keys()
    import_plugin('metrics', 'pearsonr')
    assert 'pearsonr' in metric_registry.keys()
