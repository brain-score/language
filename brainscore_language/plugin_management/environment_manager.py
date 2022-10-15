from pathlib import Path
import subprocess
import warnings


class EnvironmentManager:
    """ Runs plugins in conda environments """

    def __init__(self, runtype, plugin_ids=None):
        self.runtype = runtype
        self.plugin_ids = plugin_ids
        self.envs_dir = Path(self.get_conda_base()) / 'envs'
        self.results = None

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

    def _run_tests(self) -> int:
        """ 
        calls bash script to create conda environment, then
        runs all tests or selected test for specified plugin
        requires "test.py" file in plugin directory 
        """
        import pytest_check as check

        assert (self.plugin_directory / 'test.py').is_file(), "'test.py' not found"

        completed_process = subprocess.run(f"bash {Path(__file__).parent}/test_plugin.sh \
            {self.plugin_directory} {self.plugin_name} \
            {str(self.has_requirements).lower()} {self.test}", shell=True)
        check.equal(completed_process.returncode, 0)
        self.results[self.plugin_name] = (completed_process.returncode)

        return completed_process

    def _score_model(self) -> int:
        """ 
        calls bash script to create conda environment, then
        hands execution back to score a model on a benchmark
        """
        self.model = self.plugin_ids['model']
        self.benchmark = self.plugin_ids['benchmark']
        self.env_name = f'{self.model}_{self.benchmark}'
        self.plugin_env_path = self.envs_dir / self.env_name

        completed_process = subprocess.run(f"bash {Path(__file__).parent}/score_model.sh \
                {self.model} {self.benchmark} {self.env_name}", shell=True)

        return completed_process

    def run(self, runtype):
        """ create env for either testing or scoring """
        if runtype == 'test':
            self._run_tests()
        elif runtype == 'score':
            self._score_model()
        else:
            assert False; f'runtype {runtype} not recognized. Must be either "test" or "score".'

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
