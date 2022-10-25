import pytest
import shutil
import subprocess
import textwrap
from pathlib import Path

from brainscore_language.plugin_management.test_plugins import PluginTestRunner

DUMMY_PLUGIN = "dummy_plugin"
DUMMY_PLUGIN_PATH = Path(__file__).parent / DUMMY_PLUGIN
DUMMY_TYPE = Path(__file__).parent.name
DUMMY_TESTFILE = DUMMY_PLUGIN_PATH / "test.py"
DUMMY_REQUIREMENTS = DUMMY_PLUGIN_PATH / "requirements.txt"
DUMMY_RESULTS = {}


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
        plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH, DUMMY_RESULTS)
        assert plugin_test_runner.plugin_name == DUMMY_TYPE + '_' + DUMMY_PLUGIN

    def test_has_testfile(self):
        DUMMY_TESTFILE.unlink()
        plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH, DUMMY_RESULTS)
        with pytest.raises(Exception):
            plugin_test_runner.validate_plugin()

    def test_has_requirements(self):
        DUMMY_REQUIREMENTS.unlink()
        plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH, DUMMY_RESULTS)
        assert plugin_test_runner.has_requirements == False

    def test_run_tests(self):
        plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH, DUMMY_RESULTS)
        plugin_test_runner.run_tests()
        assert plugin_test_runner.results[plugin_test_runner.plugin_name] == 0
