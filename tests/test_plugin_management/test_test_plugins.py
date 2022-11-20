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
DUMMY_ENV_YML = DUMMY_PLUGIN_PATH / "environment.yml"
DUMMY_RESULTS = {}


class TestPluginTestRunner:
    def setup_method(self):
        DUMMY_PLUGIN_PATH.mkdir(parents=True, exist_ok=True)

        with open(DUMMY_TESTFILE, 'w') as f:
            f.write(textwrap.dedent('''\
                def test():
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

    def test_run_tests(self):
        plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH, DUMMY_RESULTS)
        plugin_test_runner.run_tests()
        assert plugin_test_runner.results[plugin_test_runner.plugin_name] == 0

    @pytest.mark.travis_slow
    def test_run_tests_with_r(self):
        with open(DUMMY_TESTFILE, 'w') as f:
            f.write(textwrap.dedent('''\
                import rpy2
                from rpy2.robjects.packages import importr

                def test_r():
                    base = importr('base')
                    assert True       
            '''))
        with open(DUMMY_ENV_YML, 'w') as f:
            f.write(textwrap.dedent('''\
                channels:
                    - conda-forge
                dependencies:
                    - r-base    
            '''))
        with open(DUMMY_REQUIREMENTS, 'w') as f:
            f.write(textwrap.dedent('''\
                rpy2    
            '''))

        plugin_test_runner = PluginTestRunner(DUMMY_PLUGIN_PATH, DUMMY_RESULTS)
        plugin_test_runner.run_tests()
        assert plugin_test_runner.results[plugin_test_runner.plugin_name] == 0
