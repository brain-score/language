import re
import sys
from pathlib import Path
from typing import List, Tuple

PLUGIN_DIRS = ['models', 'benchmarks', 'data', 'metrics']


def get_changed_files(changed_files: str) -> Tuple[List[str], List[str]]:
    changed_files_list = changed_files.split()

    plugin_files_changed = []
    non_plugin_files_changed = []

    for f in changed_files_list:
        if not any(plugin_dir in f for plugin_dir in PLUGIN_DIRS):
            non_plugin_files_changed.append(f)
        else:
            plugin_files_changed.append(f)

    return plugin_files_changed, non_plugin_files_changed


def is_plugin_only(plugins_dict: dict, non_plugin_files_changed: List[str]):
    """ 
    Stores `plugins_dict['is_plugin_only']` `"True"` or `"False"` 
    depending on whether there are any changed non-plugin files.
    """
    plugin_only = len(non_plugin_files_changed) > 0
    plugins_dict["is_plugin_only"] = "False" if plugin_only else "True"


def _get_registered_plugins(plugin_type: str, plugin_dirs: List[str]) -> List[str]:
    """
    Searches all `plugin_type` __init.py__ files for registered plugins.
    Returns list of identifiers for each registered plugin.
    """
    registered_plugins = []

    plugin_type_dir = Path(f'brainscore_language/{plugin_type}')

    for plugin_dirname in plugin_dirs:
        plugin_dirpath = plugin_type_dir / plugin_dirname
        init_file = plugin_dirpath / "__init__.py"
        with open(init_file) as f:
            registry_name = plugin_type.strip(
                's') + '_registry'  # remove plural and determine variable name, e.g. "models" -> "model_registry"
            plugin_registrations = [line for line in f if f"{registry_name}["
                                    in line.replace('\"', '\'')]
            for line in plugin_registrations:
                result = re.search(f'{registry_name}\[.*\]', line)
                identifier = result.group(0)[len(registry_name) + 2:-2] # remove brackets and quotes
                registered_plugins.append(identifier)

    return registered_plugins


def plugins_to_score(plugins_dict, plugin_files_changed) -> str:
    plugins_dict["run_score"] = "False"

    scoring_plugin_types = ("models", "benchmarks")
    scoring_plugin_paths = tuple(f'brainscore_language/{plugin_type}/' for plugin_type in scoring_plugin_types)
    model_and_benchmark_files = [fname for fname in plugin_files_changed if fname.startswith(scoring_plugin_paths)]
    if len(model_and_benchmark_files) > 0:
        plugins_dict["run_score"] = "True"
        for plugin_type in scoring_plugin_types:
            plugin_dirs = set(
                [plugin_name_from_path(fname) for fname in model_and_benchmark_files if f'/{plugin_type}/' in fname])
            plugins_to_score = _get_registered_plugins(plugin_type, plugin_dirs)
            plugins_dict[f'new_{plugin_type}'] = ' '.join(plugins_to_score)


def serialize_dict(plugins_dict: dict) -> str:
    return str(plugins_dict).replace('\'', '\"')


def plugin_name_from_path(path_relative_to_library: str) -> str:
    """
    Returns the name of the plugin from the given path. 
    E.g. `plugin_name_from_path("brainscore_vision/models/mymodel")` will return `"mymodel"`.
    """
    return path_relative_to_library.split('/')[2]


if __name__ == '__main__':
    changed_files = sys.argv[1]
    plugin_files_changed, non_plugin_files_changed = get_changed_files(changed_files)

    plugins_dict = {}
    is_plugin_only(plugins_dict, non_plugin_files_changed)
    plugins_to_score(plugins_dict, plugin_files_changed)
    plugins_dict = serialize_dict(plugins_dict)

    print(plugins_dict)
