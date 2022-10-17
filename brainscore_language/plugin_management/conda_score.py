from pathlib import Path
import pickle

from brainscore_core.metrics import Score
from brainscore_language.plugin_management.environment_manager import EnvironmentManager

SCORE_PATH = 'brainscore_language/score.pkl'

class CondaScore(EnvironmentManager):
    """ run scoring in conda environment """
    def __init__(self, plugin_ids):
        super(CondaScore, self).__init__()

        self.model = plugin_ids['model']
        self.benchmark = plugin_ids['benchmark']
        self.env_name = f'{self.model}_{self.benchmark}'
        self.script_path = f'{Path(__file__).parent}/conda_score.sh'

    def __init__():
        self.score_in_env()

    def score_in_env(self):
        """ 
        calls bash script to create conda environment, then
        hands execution back to score()
        """
        run_command = f"bash {self.script_path} \
                {self.model} {self.benchmark} {self.env_name}"

        completed_process = self.run_in_env(run_command)
        assert completed_process.returncode == 0


def save_score(score: Score):
    with open(SCORE_PATH, 'wb') as f:
        pickle.dump(score, f, pickle.HIGHEST_PROTOCOL)

def get_score():
    with open(SCORE_PATH, 'rb') as f:
        return pickle.load(f)
