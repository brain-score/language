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

	def __call__(self):
		self.validate_plugin()
		self.run_tests()
		self.teardown()

	def validate_plugin(self):
		assert (self.plugin_directory / 'test.py').is_file(), "'test.py' not found"

	def run_tests(self):
		completed_process = subprocess.run(f"./brainscore_language/plugins/create_env.sh \
			{self.plugin_directory} {self.plugin_name} \
			{str(self.has_requirements).lower()}", shell=True)
		return completed_process

	def teardown(self):
		completed_process = subprocess.run(f"conda env remove -n {self.plugin_name}", shell=True)
		return completed_process


if __name__ == '__main__':
	# run tests for each plugin in "plugins" directory
	# requires test file ("test.py")
	for plugin_directory in PLUGINS_DIRPATH.glob('[!._]*'):
		if plugin_directory.is_dir():
			plugin_test_runner = PluginTestRunner(plugin_directory)
			plugin_test_runner()
