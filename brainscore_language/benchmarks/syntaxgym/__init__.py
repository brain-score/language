from pathlib import Path
from typing import Dict, Tuple, List

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language import load_metric, benchmark_registry
from brainscore_language.benchmarks.syntaxgym.sg_suite import _load_suite

class SyntaxGymTSE(BenchmarkBase):
# A benchmark to perform SyntaxGym Targeted Syntactic Evaluations (TSE).
# See the SyntaxGym website for information about structuring test_suites:
# https://cpllab.github.io/syntaxgym-core/architecture.html
    def __init__(self):
        super(SyntaxGymTSE, self).__init__(
            identifier='syntaxgym',
            version=1,
            parent='engineering',
            ceiling=None,
            bibtex=None)
        self.metric = load_metric('accuracy')
        self.data = _load_suite(Path(__file__).parent / 'test_suite.json')

    def _get_region_totals(self, candidate: ArtificialSubject
                           ) -> Dict[Tuple[str, int], float]:
        """
        Compute region-level surprisal totals for the given subject.
        """
        raise NotImplementedError()

    def _evaluate_predictions(self, region_totals: Dict[Tuple[str, int], float]
                              ) -> List[List[bool]]:
        """
        Compute prediction results for each item.
        """
        raise NotImplementedError()

    def __call__(self, candidate: ArtificialSubject)-> Score:
        suite_regions = list(self.data.iter_regions())
        candidate.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        region_totals = {}
        predictions = []
        item_dict_plus_results = []
        # SyntaxGym logic wrapper around digest_text
        for item_num, item in enumerate(self.data.items):
            for condition_num, condition in enumerate(self.data.condition_names):
                text = suite_regions[item_num * len(self.data.condition_names) + condition_num]
                surprisals = candidate.digest_text(text)['behavior']
                for i, region in enumerate(self.data.region_names):
                    region_totals[(condition, i + 1)] = surprisals[i].values
            for pred in self.data.predictions:
                item_pred_results = pred.apply_prediction_formula(region_totals)
                predictions.append(item_pred_results)
            item_dict_plus_results.append([region_totals, predictions])
       # The final score is the percentage of predictions that are "True"
        targets = [True] * len(predictions)
        score = self.metric(predictions, targets)
        return score

benchmark_registry['syntaxgym'] = SyntaxGymTSE