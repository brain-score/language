import os
from pathlib import Path
import pytest
import subprocess

PLUGINS_DIRPATH = Path(__file__).parent


class PluginTestRunner:
	def __init__(self, plugin_directory):
		self.plugin_directory = plugin_directory
		self.plugin_name = str(self.plugin_directory).split('/')[-1]

	def __call__(self):
		self.run_tests()
		self.teardown()

	def run_tests(self):
		subprocess.run(f"./brainscore_language/plugins/create_env.sh \
			{self.plugin_path} {self.plugin_name}", shell=True)

	def teardown(self):
		subprocess.run(f"conda env remove -n {self.plugin_name}", shell=True)


if __name__ == '__main__':
	# run tests for each plugin in "plugins" directory
	# requires test file ("test.py") and requirements file ("requirements.txt")
	for plugin_directory in PLUGINS_DIRPATH.glob('[!._]*'):
		if plugin_directory.is_dir():
			assert (plugin_directory / 'test.py').is_file(), "'test.py' not found"
			assert (plugin_directory / 'requirements.txt').is_file(), "'requirements.txt' not found"
			PluginTestRunner(plugin_directory)