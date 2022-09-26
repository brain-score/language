import argparse
import pytest_check as check
import shutil
import subprocess
import warnings
from pathlib import Path

PLUGIN_TYPES = ['benchmarks', 'data', 'metrics', 'models']


class PluginTestRunner:
    def __init__(self, plugin_directory, results, test=False):
        self.plugin_directory = plugin_directory
        self.plugin_type = Path(self.plugin_directory).parent.name
        self.plugin_name = self.plugin_type + '_' + Path(self.plugin_directory).name
        self.plugin_env_path = Path(self.get_conda_base()) / 'envs' / self.plugin_name
        self.has_requirements = (self.plugin_directory / 'requirements.txt').is_file()
        self.test = test if test else False
        self.results = results

    def __call__(self):
        self.validate_plugin()
        self.run_tests()
        self.teardown()

    def get_conda_base(self):
        return subprocess.check_output("conda info --base", shell=True).strip().decode('utf-8')

    def validate_plugin(self):
        assert (self.plugin_directory / 'test.py').is_file(), "'test.py' not found"

    def run_tests(self):
        completed_process = subprocess.run(f"bash {Path(__file__).parent}/run_plugin.sh \
			{self.plugin_directory} {self.plugin_name} \
			{str(self.has_requirements).lower()} {self.test}", shell=True)
        check.equal(completed_process.returncode, 0)
        self.results[self.plugin_name] = (completed_process.returncode)

        return completed_process

    def teardown(self):
        completed_process = subprocess.run(f"output=`conda env remove -n {self.plugin_name} 2>&1` || echo $output",
                                           shell=True)
        if completed_process.returncode != 0:  # directly remove env dir if conda fails
            try:
                shutil.rmtree(self.plugin_env_path)
                completed_process.returncode = 0
            except Exception as e:
                warnings.warn(f"conda env {self.plugin_name} removal failed and must be manually deleted.")
        return completed_process

def arg_parser():
    parser = argparse.ArgumentParser(description='Run single specified test or tests for each plugin')
    parser.add_argument('test_file', type=str, nargs='?',help='Optional: full path of test file')
    parser.add_argument('--test', type=str, help='Optional: name of test to run', required=False)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # requires test file ("test.py")
    results = {}

    args = arg_parser()
    if args.test_file and Path(args.test_file).exists():
        filename = args.test_file.split('/')[-1]
        plugin_dirname = args.test_file.split('/')[-2]
        plugin_type = args.test_file.split('/')[-3]
        plugins_dir = Path(Path(__file__).parents[1], plugin_type)
        plugin = plugins_dir / plugin_dirname
        assert filename == "test.py", "Filepath not recognized as test file, must be 'test.py'."
        assert plugin_type in PLUGIN_TYPES, "Filepath not recognized as plugin test file."
        plugin_test_runner = PluginTestRunner(plugin, results, test=args.test)
        plugin_test_runner()
    elif not args.file:
        for plugin_type in PLUGIN_TYPES:
            plugins_dir = Path(Path(__file__).parents[1], plugin_type)
            for plugin in plugins_dir.glob('[!._]*'):
                if plugin.is_dir():
                    plugin_test_runner = PluginTestRunner(plugin, results)
                    plugin_test_runner()
    else:
        warnings.warn("Filepath does not exist or is not recognized as plugin test file.")

    # plugins_with_errors = {k:v for k,v in results.items() if v == 1}
    # num_plugins_failed = len(plugins_with_errors)
    # assert num_plugins_failed == 0, f"\n{num_plugins_failed} plugin tests failed\n{plugins_with_errors}"
