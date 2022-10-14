import argparse
import shutil
from typing import Dict, List
import warnings
from pathlib import Path

from environment_manager import EnvironmentManager

PLUGIN_TYPES = ['benchmarks', 'data', 'metrics', 'models']


class PluginTestRunner(EnvironmentManager):
    """Runs plugin tests (requires "test.py" for each plugin)
    
    Usage examples:

    # Run all tests for futrell2018 benchmark:
    python brainscore_language/plugin_management/test_plugins.py brainscore_language/benchmarks/futrell2018/test.py

    # Run only tests with names matching specified pattern (test_exact):
    python brainscore_language/plugin_management/test_plugins.py brainscore_language/benchmarks/futrell2018/test.py --test=test_exact

    # Run all tests for all plugins:
    python brainscore_language/plugin_management/test_plugins.py 

    """
    def __init__(self, plugin_directory:Path, results:Dict, test=False):
        EnvironmentManager.__init__(self, 'test')

        self.plugin_directory = plugin_directory
        self.plugin_type = Path(self.plugin_directory).parent.name
        self.plugin_name = self.plugin_type + '_' + Path(self.plugin_directory).name
        self.plugin_env_path = self.envs_dir / self.plugin_name
        self.has_requirements = (self.plugin_directory / 'requirements.txt').is_file()
        self.test = test if test else False
        self.results = results


def arg_parser() -> List[str]:
    parser = argparse.ArgumentParser(description='Run single specified test or all tests for each plugin')
    parser.add_argument('test_file', type=str, nargs='?',help='Optional: path of target test file')
    parser.add_argument('--test', type=str, help='Optional: name of test to run', required=False)
    args = parser.parse_args()

    return args

def run_specified_tests(args:List[str], results:Dict):
    """ Runs either a single test or all tests in a specified test.py """
    filename = args.test_file.split('/')[-1]
    plugin_dirname = args.test_file.split('/')[-2]
    plugin_type = args.test_file.split('/')[-3]
    plugin = Path(Path(__file__).parents[1], plugin_type) / plugin_dirname
    assert filename == "test.py", "Filepath not recognized as test file, must be 'test.py'."
    assert plugin_type in PLUGIN_TYPES, "Filepath not recognized as plugin test file."
    plugin_test_runner = PluginTestRunner(plugin, results, test=args.test)
    plugin_test_runner()

def run_all_tests(results:Dict):
    """ Runs tests for all plugins """
    for plugin_type in PLUGIN_TYPES:
        plugins_dir = Path(Path(__file__).parents[1], plugin_type)
        for plugin in plugins_dir.glob('[!._]*'):
            if plugin.is_dir():
                plugin_test_runner = PluginTestRunner(plugin, results)
                plugin_test_runner()


if __name__ == '__main__':
    results = {}

    args = arg_parser()
    if not args.test_file:
        run_all_tests(results)
    elif args.test_file and Path(args.test_file).exists():
        run_specified_tests(args, results)
    else:
        warnings.warn("Test file not found.")

    plugins_with_errors = {k:v for k,v in results.items() if v == 1}
    num_plugins_failed = len(plugins_with_errors)
    assert num_plugins_failed == 0, f"\n{num_plugins_failed} plugin tests failed\n{plugins_with_errors}"
