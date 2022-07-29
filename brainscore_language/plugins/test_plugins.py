import os
from pathlib import Path
import pytest
import subprocess

PLUGINS_DIRPATH = Path(__file__).parent


class PluginTestRunner:
	def __init__(self, plugin_directory):
		self.plugin_directory = plugin_directory
		self.plugin_name = str(self.plugin_directory).split('/')[-1]
		self.has_requirements = (self.plugin_directory / 'requirements.txt').is_file()
		self.conda_envs_dirpath = Path('/'.join(os.environ["CONDA_PREFIX"].split('/')[:-1]))
		self.plugin_env_path = self.conda_envs_dirpath / self.plugin_name

	def __call__(self):
		self.validate_plugin()
		self.run_tests()
		self.teardown()

	def validate_plugin(self):
		assert (self.plugin_directory / 'test.py').is_file(), "'test.py' not found"

	def run_tests(self):
		return subprocess.run(f"./brainscore_language/plugins/create_env.sh \
			{self.plugin_directory} {self.plugin_name} \
			{str(self.has_requirements).lower()}", shell=True)

	def teardown(self):
		completed_process = subprocess.run(f"conda env remove -n {self.plugin_name}", shell=True)
		if completed_process.returncode is not 0:
			subprocess.run(f"rm -rf {self.plugin_env_path}")
		return completed_process


if __name__ == '__main__':
	# run tests for each plugin in "plugins" directory
	# requires test file ("test.py")
	for plugin_directory in PLUGINS_DIRPATH.glob('[!._]*'):
		if plugin_directory.is_dir():
			plugin_test_runner = PluginTestRunner(plugin_directory)
			plugin_test_runner()
