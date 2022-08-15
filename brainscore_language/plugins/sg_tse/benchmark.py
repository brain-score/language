import logging
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_language import load_metric, benchmarks
from brainscore_language.models.huggingface import HuggingfaceSubject
from brainscore_language.plugins.sg_tse import compute_surprisals, evaluate

logger = logging.getLogger(__name__)

class SyntaxGymTargetedSyntacticEval(BenchmarkBase):
    def __init__(self):
        super(SyntaxGymTargetedSyntacticEval, self).__init__(
            identifier='SG-TSE',
            version=1,
            parent='engineering',
            ceiling_func=None,
            bibtex=None)

    def __call__(self, candidate: HuggingfaceSubject):
        suite = compute_surprisals(candidate, 'test_suite.json')
        predictions = evaluate(suite)
        targets = [True] * len(predictions)
        metric = load_metric('accuracy')
        score = metric(predictions, targets)
        return score


benchmarks['SG-TSE'] = SyntaxGymTargetedSyntacticEval
