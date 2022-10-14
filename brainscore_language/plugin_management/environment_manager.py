from pathlib import Path
import subprocess


class EnvironmentManager:
    """ Runs plugins in conda environments """

    def __init__(self, runtype, identifiers=None):
        self.runtype = runtype
        self.identifiers = identifiers
        self.envs_dir = Path(self.get_conda_base()) / 'envs'
        self.results = None

    def __call__(self):
        self.run(self.runtype)
        self.teardown()

    def get_conda_base(self):
        return subprocess.check_output("conda info --base", shell=True).strip().decode('utf-8')

    def _run_tests(self):
        import pytest_check as check

        assert (self.plugin_directory / 'test.py').is_file(), "'test.py' not found"

        completed_process = subprocess.run(f"bash {Path(__file__).parent}/test_plugin.sh \
            {self.plugin_directory} {self.plugin_name} \
            {str(self.has_requirements).lower()} {self.test}", shell=True)
        check.equal(completed_process.returncode, 0)
        self.results[self.plugin_name] = (completed_process.returncode)

        return completed_process

    def _score_model(self):
        self.model = self.identifiers['model']
        self.benchmark = self.identifiers['benchmark']

        score = subprocess.check_output(f"bash {Path(__file__).parent}/score_model.sh \
            {self.model} {self.benchmark}", shell=True)
        self.results = (score)
        print(score)

        return completed_process

    def run(self, runtype):
        if runtype == 'test':
            self._run_tests()
        elif runtype == 'score':
            self._score_model()
        else:
            assert False; f'runtype {runtype} not recognized. Must be either "test" or "score".'

    def teardown(self):
        completed_process = subprocess.run(f"output=`conda env remove -n {self.plugin_name} 2>&1` || echo $output",
                                           shell=True)
        if completed_process.returncode != 0:  # directly remove env dir if conda fails
            try:
                shutil.rmtree(self.plugin_env_path)
                completed_process.returncode = 0
            except Exception as e:
                warnings.warn(f"conda env {self.plugin_name} removal failed and must be manually deleted.")
        return completed_process
