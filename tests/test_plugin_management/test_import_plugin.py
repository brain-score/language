import os

from brainscore_language import benchmark_registry, metric_registry
from brainscore_language.plugin_management.import_plugin import import_plugin


def test_import_plugin():
    if 'Wikitext-accuracy' in benchmark_registry.keys():
        benchmark_registry['Wikitext-accuracy'].pop()
    assert 'Wikitext-accuracy' not in benchmark_registry.keys()
    import_plugin('benchmarks', 'Wikitext-accuracy')
    assert 'Wikitext-accuracy' in benchmark_registry.keys()


def test_no_installation():
    os.environ['BS_INSTALL_DEPENDENCIES'] = 'no'
    try:
        import_plugin('data', 'wikitext-2/test')
    except Exception as e:
        assert "No module named 'datasets'" in str(e)
