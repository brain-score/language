from pathlib import Path
import json
from typing import Dict, Tuple, List

import numpy as np
import requests

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language import load_metric, benchmark_registry
from brainscore_language.benchmarks.syntaxgym.sg_suite import _load_suite, Suite

# This is a Brain-Score Language benchmark to perform SyntaxGym Targeted Syntactic Evaluations (TSE).
#
# See the SyntaxGym website for information about structuring test_suites:
# https://cpllab.github.io/syntaxgym-core/architecture.html
#
# The following are options for loading test_suites:
# EXAMPLE #1: (Loading a single test_suite) suite_list = [Path(__file__).parent / 'test_suite.json']
# EXAMPLE #2: (Loading multiple test_suites as a list): suite_list = [Path(__file__).parent / 'test_suite2.json', Path(__file__).parent / 'test_suite3.json']
# EXAMPLE #3: (Using a URL to load a test_suite directly from GitHub): suite_list = ['https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/center_embed.json']
# EXAMPLE #4: (Loading one of the test suites in test_suites.py/test_suites.json.  This method also works for multiple suites.):
#               with open(Path(__file__).parent / 'test_suites.json') as json_file:
#                   test_suite_dict = json.load(json_file)
#                   suite_list = [test_suite_dict['center_embed']]


def SyntaxGym2020():
    with open(Path(__file__).parent / 'test_suites.json') as json_file:
      test_suite_dict = json.load(json_file)
      suite_paths = list(test_suite_dict.values())
    return SyntaxGymTSE(suite_paths)


class SyntaxGymTSE(BenchmarkBase):
    def __init__(self, suite_ref_list):
        super(SyntaxGymTSE, self).__init__(
            identifier='syntaxgym-2020',
            version=1,
            parent='engineering',
            ceiling=None,
            bibtex=None)

        self.sub_benchmarks = [
            SyntaxGymSingleTSE(suite_ref) for suite_ref in suite_ref_list]

    def __call__(self, candidate: ArtificialSubject) -> Score:
        final_score = Score()
        final_score.values = np.mean([sub_benchmark(candidate) for sub_benchmark in self.sub_benchmarks])
        return final_score

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
        self.suite = self._load_suite(suite_ref)

    def _load_suite(self, suite_ref):
        if str(suite_ref)[0:5] == "https":
            suite = requests.get(suite_ref).json()
            return Suite.from_dict(suite)
        if isinstance(suite_ref, (str, Path)):
            suite_ref = Path(suite_ref)
            if not suite_ref.exists():
                # Specify relative to package path.
                suite_ref = Path(__file__).parent / "suites" / "syntaxgym-2020" / suite_ref
                if not suite_ref.exists():
                    # Try adding extension
                    suite_ref = suite_ref.with_suffix(".json")
                    if not suite_ref.exists():
                        raise FileNotFoundError(f'Could not find suite at {suite_ref}')

        return _load_suite(suite_ref)

    def get_region_totals(self, candidate: ArtificialSubject
                          ) -> List[Dict[Tuple[str, int], float]]:
        """
        Compute region-level surprisal totals for the given subject.
        """
        suite_regions = list(self.suite.iter_regions())
        candidate.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        region_totals = []

        # SyntaxGym logic wrapper around digest_text
        for item_num, item in enumerate(self.suite.items):
            region_totals_i = {}
            for condition_num, condition in enumerate(self.suite.condition_names):
                text = suite_regions[item_num * len(self.suite.condition_names) + condition_num]
                surprisals = candidate.digest_text(text)['behavior']
                for i, region in enumerate(self.suite.region_names):
                    surprisal_i = surprisals[i].values
                    if np.isnan(surprisal_i):
                        surprisal_i = 0.
                    region_totals_i[condition, i + 1] = surprisal_i

            region_totals.append(region_totals_i)
        
        return region_totals

    def evaluate_predictions(self, region_totals: List[Dict[Tuple[str, int], float]]
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
        region_totals = self.get_region_totals(candidate)
        prediction_results = self.evaluate_predictions(region_totals)

        # Compute conjunction of all predictions within-item.
        conj_predictions = np.array(prediction_results).all(axis=1)
        targets = [True] * len(conj_predictions)
        score = self.metric(conj_predictions, targets)

        return score

benchmark_registry["syntaxgym-2020"] = SyntaxGym2020