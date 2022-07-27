import os
import pytest
import subprocess

PLUGINS_DIRPATH = "brainscore_language/plugins"


class PluginTestRunner:
	def __init__(self, plugin_path, plugin_name):
		self.plugin_path = plugin_path
		self.plugin_name = plugin_name
		self.run_tests()
		self.teardown()

	def run_tests(self):
		subprocess.run("./brainscore_language/plugins/create_env.sh \
			{plugin_path} {plugin_name}".format(
			plugin_path=self.plugin_path, plugin_name=self.plugin_name), shell=True)

	def teardown(self):
		subprocess.run("conda env remove -n {plugin_name}".format(
			plugin_name=self.plugin_name), shell=True)


# run tests for each plugin in "plugins" directory
# requires test file ("test.py") and requirements file ("requirements.txt")
for filename in os.listdir(PLUGINS_DIRPATH):
	f = os.path.join(PLUGINS_DIRPATH, filename)
	if os.path.isdir(f) and filename[0] not in ["_", "."]:
		assert(os.path.exists(os.path.join(f, 'test.py'))), "'test.py' not found"
		assert(os.path.exists(os.path.join(f, 'requirements.txt'))), "'requirements.txt' not found"
		PluginTestRunner(f, filename)