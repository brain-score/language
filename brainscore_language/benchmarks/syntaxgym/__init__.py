from pathlib import Path
import statistics
import numpy as np
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
    def __init__(self, suite_ref_list):
        super(SyntaxGymTSE, self).__init__(
            identifier='syntaxgym',
            version=1,
            parent='engineering',
            ceiling=None,
            bibtex=None)

        self.sub_benchmarks = [
            SyntaxGymSingleTSE(suite_ref) for suite_ref in suite_ref_list]

    def __call__(self, candidate: ArtificialSubject) -> Score:
        return np.mean([
            sub_benchmark(candidate) for sub_benchmark in self.sub_benchmarks
        ])


class SyntaxGymSingleTSE(BenchmarkBase):
    def __init__(self, suite_ref):
        super(SyntaxGymSingleTSE, self).__init__(
            identifier='syntaxgym-single',
            version=1,
            parent='engineering',
            ceiling=None,
            bibtex=None)

        self.metric = load_metric('accuracy')
        # TODO support non-path ref
        self.suite = _load_suite(suite_ref)

    def _get_region_totals(self, candidate: ArtificialSubject
                           ) -> Dict[Tuple[str, int], float]:
        """
        Compute region-level surprisal totals for the given subject.
        """
        suite_regions = list(self.suite.iter_regions())
        candidate.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        region_totals = {}

        # SyntaxGym logic wrapper around digest_text
        for item_num, item in enumerate(self.suite.items):
            for condition_num, condition in enumerate(self.suite.condition_names):
                text = suite_regions[item_num * len(self.suite.condition_names) + condition_num]
                surprisals = candidate.digest_text(text)['behavior']
                for i, region in enumerate(self.suite.region_names):
                    region_totals[(condition, i + 1)] = surprisals[i].values
        
        return region_totals

    def _evaluate_predictions(self, region_totals: List[Dict[Tuple[str, int], float]]
                              ) -> List[List[bool]]:
        """
        Compute prediction results for each item.
        """
        prediction_results = []
        for item_region_totals in region_totals:
            prediction_results.append([
                pred.apply_prediction_formula(item_region_totals)
                for pred in self.suite.predictions
            ])

        return prediction_results

    def __call__(self, candidate: ArtificialSubject)-> Score:
        region_totals = self._get_region_totals(candidate)
        prediction_results = self._evaluate_predictions([region_totals])

        # Compute conjunction of all predictions within-item.
        conj_predictions = np.array(prediction_results).all(axis=1)
        targets = [True] * len(conj_predictions)
        score = self.metric(conj_predictions, targets)

        return score

benchmark_registry['syntaxgym'] = SyntaxGymTSE