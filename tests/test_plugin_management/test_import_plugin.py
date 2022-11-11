import importlib
import os
from pathlib import Path
import shutil
import subprocess
import textwrap

from brainscore_language import model_registry
from brainscore_language.plugin_management.import_plugin import import_plugin

dummy_model_dirpath = Path(f"brainscore_language/models/dummy_model")
current_dependencies_pref = os.getenv('BS_INSTALL_DEPENDENCIES')


class TestImportPlugin:
    def setup_method(self):
        dummy_model = dummy_model_dirpath / "model.py"
        dummy_testfile = dummy_model_dirpath / "test.py"
        dummy_requirements = dummy_model_dirpath / "test.py"
        dummy_init = dummy_model_dirpath / "__init__.py"

        dummy_model_dirpath.mkdir(parents=True, exist_ok=True)
        Path(dummy_model).touch()
        with open(dummy_model, 'w') as f:
            f.write(textwrap.dedent('''\
            class dummyModel:
                pass       
            '''))
        Path(dummy_testfile).touch()
        with open(dummy_testfile, 'w') as f:
            f.write(textwrap.dedent('''\
            def test_dummy():
                assert True        
            '''))
        Path(dummy_requirements).touch()
        with open(dummy_requirements, 'w') as f:
            f.write(textwrap.dedent('''\
            pyaztro     
            '''))
        Path(dummy_init).touch()
        with open(dummy_init, 'w') as f:
            f.write(textwrap.dedent('''\
            from brainscore_language import model_registry
            from .model import dummyModel

            model_registry['dummy-model'] = dummyModel     
            '''))

    def teardown_method(self):
        if 'dummy-model' in model_registry:
            del model_registry['dummy-model']
        subprocess.run('pip uninstall pyaztro', shell=True)
        shutil.rmtree(dummy_model_dirpath)
        if current_dependencies_pref:
            os.environ['BS_INSTALL_DEPENDENCIES'] = current_dependencies_pref

    def test_yes_dependency_installation(self):
        os.environ['BS_INSTALL_DEPENDENCIES'] = 'yes'
        assert 'dummy-model' not in model_registry
        import_plugin('models', 'dummy-model')
        assert 'dummy-model' in model_registry

    def test_no_dependency_installation(self):
        os.environ['BS_INSTALL_DEPENDENCIES'] = 'no'
        assert 'dummy-model' not in model_registry
        try:
            print("importing plugin")
            import_plugin('models', 'dummy-model')
        except Exception as e:
            assert "No module named 'pyaztro'" in str(e)
