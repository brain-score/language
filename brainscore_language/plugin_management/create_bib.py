from pathlib import Path

from brainscore_language import benchmark_registry, data_registry, metric_registry, model_registry
from brainscore_language import load_benchmark, load_dataset, load_metric, load_model

PLUGIN_TYPES = ['benchmarks', 'data', 'metrics', 'models']

def init_plugins():
    for plugin_type in PLUGIN_TYPES:
        plugins_dir = Path(Path(__file__).parents[1], plugin_type)
        for plugin in plugins_dir.glob('[!._]*'):
            if plugin.is_dir():
                __import__(str(plugin).replace('/', '.'))

def register_all_plugins(reg_types):

    registry_names = [reg_type + '_registry' for reg_type in reg_types]
    registries = {k:globals()[k] for k in globals().keys() if k in registry_names}

    return registries

def load_bibtex(registries, reg_types):
    for registry in registries:
        print(registry, list(registries[registry].keys()))

    

if __name__ == '__main__':
    # add each BibTeX to Sphinx refs.bib
    reg_types = [plugin_type.strip('s') for plugin_type in PLUGIN_TYPES]

    init_plugins()
    registries = register_all_plugins(reg_types)
    load_bibtex(registries, reg_types)