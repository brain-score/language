import subprocess

from brainscore_language.plugin_management.environment_manager import EnvironmentManager

def test_get_conda_base():
	environment_manager = EnvironmentManager()
	conda_base = environment_manager.get_conda_base()
	assert 'conda' in conda_base

def test_teardown():
	environment_manager = EnvironmentManager()
	subprocess.run(f"conda create -n {environment_manager.env_name} python=3.8 -y", shell=True)
	assert environment_manager.env_path.is_dir() == True
	environment_manager.teardown()
	assert environment_manager.env_path.is_dir() == False
