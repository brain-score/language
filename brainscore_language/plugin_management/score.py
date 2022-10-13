import re
from typing import Dict, List
import subprocess
import warnings

from brainscore_language import create_registry_preview

plugins_to_run: Dict[str, Dict[str, str]] = {}
""" ID and location of model, benchmark, data, and metric to score """

run_requirements: List = []
""" requirements for all plugins in plugins_to_run """


def score():
    pass


def _get_plugin_requirements():
    for plugin in plugins_to_run:
        plugin_path = plugins_to_run[plugin]['dir']
        requirements_file = plugin_path / "requirements.txt"
        if requirements_file.is_file():
            with open(requirements_file, 'r') as f:
                run_requirements.append([line.strip() for line in f])


def check_requirement_conflicts():
    _get_plugin_requirements()
    flattened_requirements = [req.split('=><', 1)[0] for l in run_requirements for req in l]
    if len(flattened_requirements) != len(set(flattened_requirements)):
        warnings.warn("Warning: duplicate requirements found. May produce unexpected results.")


def preview_benchmark(identifier: str):
    benchmark_dir = create_registry_preview('benchmarks', identifier)
    plugins_to_run['benchmark'] = {'dir': benchmark_dir, 'id': identifier}

    # get metric and data info !WILL NOT WORK IF MORE THAN ONE SPECIFIED IN INIT FILE
    plugin_type_map = {'metric':'metrics', 'dataset':'data'}
    for plugin_type in plugin_type_map:
        with open(f"{benchmark_dir}/__init__.py", 'r') as f:
            loader = f'load_{plugin_type}'
            plugin_loads = [line for line in f if f'{loader}(' in line]
            loaded_plugins = [re.findall(r'\(.*?\)', line)[0].strip('()\'') for line in plugin_loads]
            plugin_id = loaded_plugins[0] if len(loaded_plugins) == 1 else None
            assert plugin_id, f'more than one {plugin_type} found'
            plugin_dir = create_registry_preview(plugin_type_map[plugin_type], plugin_id)
            plugins_to_run[plugin_type] = {'dir': plugin_dir, 'id': plugin_id}


def preview_model(identifier: str):
    model_dir = create_registry_preview('models', identifier)
    plugins_to_run['model'] = {'dir': model_dir, 'id': identifier}


if __name__ == '__main__':
    model_identifier = 'distilgpt2'
    benchmark_identifier = 'Futrell2018-pearsonr'
    preview_model(model_identifier)
    preview_benchmark(benchmark_identifier)
    check_requirement_conflicts()
    score()