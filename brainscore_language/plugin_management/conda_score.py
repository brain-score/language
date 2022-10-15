from pathlib import Path

from brainscore_language.plugin_management.environment_manager import EnvironmentManager


class CondaScore(EnvironmentManager):
    """ run scoring in conda environment """
    def __init__(self, plugin_ids):
        EnvironmentManager.__init__(self)

        self.model = plugin_ids['model']
        self.benchmark = plugin_ids['benchmark']
        self.env_name = f'{self.model}_{self.benchmark}'
        self.plugin_env_path = self.envs_dir / self.env_name
        self.script_path = f'{Path(__file__).parent}/conda_score.sh'

    def __call__(self):
        self.score_in_env()
        self.teardown()

    def score_in_env(self) -> int:
        """ 
        calls bash script to create conda environment, then
        hands execution back to score()
        """
        run_command = f"bash {self.script_path} \
                {self.model} {self.benchmark} {self.env_name}"

        completed_process = self.run_in_env(run_command)

