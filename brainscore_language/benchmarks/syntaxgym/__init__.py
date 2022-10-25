from pathlib import Path
import statistics
import numpy
from typing import Dict, Tuple, List

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language import load_metric, benchmark_registry
from brainscore_language.benchmarks.syntaxgym.sg_suite import _load_suite


#NOTE1: The following lines are to test the benchmark's ability to run the single large case it was already running
# which it does if the second line is used.  If the first line is used, the benchmark will run both the cases in
# the list and return the average of the scores.  This second case returns the average of the scores by replacing
# the score value in the Score assembly for the final test suite in the list with the average score for all the suites.
#NOTE2: If you switch to the case with two suites in the list below (test_suite2.json and test_suite3.json),
# change the expected value in test_integration.py to 0.5.

#suite_list = [Path(__file__).parent / 'test_suite2.json', Path(__file__).parent / 'test_suite3.json']
suite_list = [Path(__file__).parent / 'test_suite.json']

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
        all_scores = []
        for suite_num, suite in enumerate(suite_list):
            self.data = _load_suite(suite)
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
            all_scores.append(score.values.tolist())
        final_score = statistics.mean(all_scores)
        score.values=numpy.array(final_score)
        return score

benchmark_registry['syntaxgym'] = SyntaxGymTSE