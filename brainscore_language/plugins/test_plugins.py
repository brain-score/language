import os
from pathlib import Path
import pytest
import shutil
import subprocess
import warnings

PLUGINS_DIRPATH = Path(__file__).parent


class PluginTestRunner:
	def __init__(self, plugin_directory):
		self.plugin_directory = plugin_directory
		self.plugin_name = Path(self.plugin_directory).name
		self.plugin_env_path = Path(self.get_conda_base()) / 'envs' / self.plugin_name
		self.has_requirements = (self.plugin_directory / 'requirements.txt').is_file()

	def __call__(self):
		self.validate_plugin()
		self.run_tests()
		self.teardown()

	def get_conda_base(self):
		return subprocess.check_output("conda info --base", shell=True).strip().decode('utf-8')

	def validate_plugin(self):
		assert (self.plugin_directory / 'test.py').is_file(), "'test.py' not found"

	def run_tests(self):
		completed_process = subprocess.run(f"./brainscore_language/plugins/create_env.sh \
			{self.plugin_directory} {self.plugin_name} \
			{str(self.has_requirements).lower()}", shell=True)
		assert completed_process.returncode == 0
		return completed_process

	def teardown(self):
		completed_process = subprocess.run(f"conda env remove -n {self.plugin_name}", shell=True)
		if completed_process.returncode != 0: # directly remove env dir if conda fails
			try:
				shutil.rmtree(plugin_env_path)
				completed_process = 0
			except Exception as e:
				warnings.warn(f"conda env {self.plugin_name} removal failed and must be manually deleted.")
		return completed_process


if __name__ == '__main__':
	# run tests for each plugin in "plugins" directory
	# requires test file ("test.py")
	for plugin_directory in PLUGINS_DIRPATH.glob('[!._]*'):
		if plugin_directory.is_dir():
			plugin_test_runner = PluginTestRunner(plugin_directory)
			plugin_test_runner()
