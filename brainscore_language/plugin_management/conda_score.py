import os
import pickle
import tempfile
from pathlib import Path

from brainscore_core.metrics import Score
from brainscore_language.plugin_management.environment_manager import EnvironmentManager

SCORE_PATH = tempfile.NamedTemporaryFile(delete=False).name


class CondaScore(EnvironmentManager):
    """ run scoring in conda environment """

    def __init__(self, model_identifier: str, benchmark_identifier: str):
        super(CondaScore, self).__init__()

        self.model = model_identifier
        self.benchmark = benchmark_identifier
        self.env_name = f'{self.model}_{self.benchmark}'
        self.script_path = f'{Path(__file__).parent}/conda_score.sh'
        self.result = self.score_in_env()
        self.score = self.read_score()

    def score_in_env(self) -> 'subprocess.CompletedProcess[bytes]':
        """ 
        calls bash script to create conda environment, then
        hands execution back to score()
        """
        run_command = f"bash {self.script_path} \
                {self.model} {self.benchmark} {self.env_name}"

        completed_process = self.run_in_env(run_command)
        assert completed_process.returncode == 0

        return completed_process

    @staticmethod
    def read_score():
        with open(SCORE_PATH, 'rb') as f:
            score = pickle.load(f)
            os.remove(SCORE_PATH)
            return score

    @staticmethod
    def save_score(score: Score):
        with open(SCORE_PATH, 'wb') as f:
            pickle.dump(score, f, pickle.HIGHEST_PROTOCOL)
