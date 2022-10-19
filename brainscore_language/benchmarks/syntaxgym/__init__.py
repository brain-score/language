from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language import load_metric, benchmark_registry
from brainscore_language.benchmarks.syntaxgym.sg_suite import _load_suite

class SyntaxGymTSE(BenchmarkBase):
    def __init__(self):
        super(SyntaxGymTSE, self).__init__(
            identifier='syntaxgym',
            version=1,
            parent='engineering',
            ceiling=None,
            bibtex=None)
        self.metric = load_metric('accuracy')
        self.data = _load_suite(Path(__file__).parent / 'test_suite.json')

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
