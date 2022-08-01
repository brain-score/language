import os
from pathlib import Path
import pytest
import shutil
import subprocess
import textwrap

from brainscore_language.plugins.test_plugins import PluginTestRunner

DUMMY_PLUGIN = "dummy_plugin"
DUMMY_PLUGIN_PATH = Path(__file__).parent / DUMMY_PLUGIN
DUMMY_TESTFILE = DUMMY_PLUGIN_PATH / "test.py"
DUMMY_REQUIREMENTS = DUMMY_PLUGIN_PATH / "requirements.txt"


class TestPluginTestRunner:
	def setup_method(self):
		DUMMY_PLUGIN_PATH.mkdir(parents=True, exist_ok=True)
		Path(DUMMY_REQUIREMENTS).touch()
		Path(DUMMY_TESTFILE).touch()
		with open(DUMMY_TESTFILE, 'w') as f:
			f.write(textwrap.dedent('''\
			def test_dummy():
				assert True        
			'''))

	def teardown_method(self):
		shutil.rmtree(DUMMY_PLUGIN_PATH)

	def test_plugin_name(self):
		plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH)
		assert plugin_test_runner.plugin_name == DUMMY_PLUGIN

	def test_get_conda_base(self):
		plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH)
		try:
        	plugin_test_runner.get_conda_base()
	    except CalledProcessError as e:
	        assert False, e

	def test_has_testfile(self):
		DUMMY_TESTFILE.unlink()
		plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH)
		with pytest.raises(Exception):
			plugin_test_runner.validate_plugin()

	def test_has_requirements(self):
		DUMMY_REQUIREMENTS.unlink()
		plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH)
		assert plugin_test_runner.has_requirements == False

	def test_run_tests(self):
		plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH)
		completed_process = plugin_test_runner.run_tests()
		assert completed_process.returncode == 0

	def test_teardown(self):
		plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH)
		subprocess.run(f"conda create -n {DUMMY_PLUGIN} python=3.8 -y", shell=True)
		assert plugin_test_runner.plugin_env_path.is_dir() == True
		plugin_test_runner.teardown()
		assert plugin_test_runner.plugin_env_path.is_dir() == False
