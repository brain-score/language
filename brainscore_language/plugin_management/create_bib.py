from pathlib import Path

from brainscore_language import benchmark_registry, data_registry, metric_registry, model_registry
from brainscore_language import load_benchmark, load_data, load_metric, load_model

PLUGIN_DIRS = ['benchmarks', 'data', 'metrics', 'models']

def init_plugins():
    for plugin_dirtype in PLUGIN_DIRS:
        plugins_dir = Path(Path(__file__).parents[1], plugin_dirtype)
        for plugin in plugins_dir.glob('[!._]*'):
            if plugin.is_dir():
                __import__(str(plugin).replace('/', '.'))

def register_all_plugins(plugin_types):
    registry_names = [plugin_type + '_registry' for plugin_type in plugin_types]
    registries = {k:globals()[k] for k in globals().keys() if k in registry_names}

    return registries

def load_bibtex(registries, plugin_types):
    loader_names = ['load_' + plugin_type]
    loaders = {k:globals()[k] for k in globals().keys() if k in loader_names}

    print(loaders)

    # for registry in registries:
    #     print(registry, list(registries[registry].keys()))

    

if __name__ == '__main__':
    # add each BibTeX to Sphinx refs.bib
    plugin_types = [plugin_dirtype.strip('s') for plugin_dirtype in PLUGIN_DIRS]

    init_plugins()
    registries = register_all_plugins(plugin_types)
    load_bibtex(registries, plugin_types)