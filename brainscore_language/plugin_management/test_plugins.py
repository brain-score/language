import pytest_check as check
import shutil
import subprocess
import warnings
from pathlib import Path

PLUGIN_TYPES = ['benchmarks', 'data', 'metrics', 'models']


class PluginTestRunner:
    def __init__(self, plugin_directory, results):
        self.plugin_directory = plugin_directory
        self.plugin_type = Path(self.plugin_directory).parent.name
        self.plugin_name = self.plugin_type + '_' + Path(self.plugin_directory).name
        self.plugin_env_path = Path(self.get_conda_base()) / 'envs' / self.plugin_name
        self.has_requirements = (self.plugin_directory / 'requirements.txt').is_file()
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
			{str(self.has_requirements).lower()}", shell=True)
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


if __name__ == '__main__':
    # run tests for each plugin
    # requires test file ("test.py")
    results = {}

    for plugin_type in PLUGIN_TYPES:
        plugins_dir = Path(Path(__file__).parents[1], plugin_type)
        for plugin in plugins_dir.glob('[!._]*'):
            if plugin.is_dir():
                plugin_test_runner = PluginTestRunner(plugin, results)
                plugin_test_runner()

    plugins_with_errors = {k:v for k,v in results.items() if v == 1}
    num_plugins_failed = len(plugins_with_errors)
    assert num_plugins_failed == 0, f"\n{num_plugins_failed} plugin tests failed\n{plugins_with_errors}"
