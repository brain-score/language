from pathlib import Path
import pytest_check as check
import subprocess
import warnings


class EnvironmentManager:
    """ Runs plugins in conda environments """

    def __init__(self):
        self.envs_dir = Path(self.get_conda_base()) / 'envs'

    def __call__(self):
        self.run(self.runtype)
        self.teardown()

    def get_conda_base(self) -> str:
        """ return location of conda directory """
        try:
            return subprocess.check_output("conda info --base", shell=True).strip().decode('utf-8')
        except Exception as e:
            warnings.warn(f"{e}. Please ensure that conda is properly installed " \
                "(https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).")

    def run_in_env(self, run_command=str) -> int:

        print(run_command)
        completed_process = subprocess.run({run_command}, shell=True)
        print(completed_process)
        check.equal(completed_process.returncode, 0)
        
        return completed_process

    def teardown(self) -> int:
        """ 
        delete conda environment after use
        shutil removal if deletion fails (not uncommon if build was interrupted)
        """
        completed_process = subprocess.run(f"output=`conda env remove -n {self.env_name} 2>&1` || echo $output",
                                           shell=True)
        if completed_process.returncode != 0:  # directly remove env dir if conda fails
            try:
                shutil.rmtree(self.plugin_env_path)
                completed_process.returncode = 0
            except Exception as e:
                warnings.warn(f"conda env {self.env_name} removal failed and must be manually deleted.")
        
        return completed_process
