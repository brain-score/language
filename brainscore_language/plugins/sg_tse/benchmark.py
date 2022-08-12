import logging
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_language import load_metric, benchmarks
from brainscore_language.models.huggingface import HuggingfaceSubject
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.plugins.sg_tse import compute_surprisals, evaluate

logger = logging.getLogger(__name__)

BIBTEX = """MIT Psycholinguistics lab; SyntaxGym, LM-Zoo; ,
  title={SyntaxGym Targeted Syntactic Evaluations},
  author={MIT Pscyho-linguistics Lab, Jim Neidhoefer},
  conference={N/A},
  url={https://syntaxgym.org/},
  year={2022}
}"""

class SyntaxGymTargetedSyntacticEval(BenchmarkBase):
    def __init__(self):
        super(SyntaxGymTargetedSyntacticEval, self).__init__(
            identifier='SG-TSE',
            version=1,
            parent='engineering',
            ceiling_func=None,
            bibtex=BIBTEX)

    def __call__(self, candidate: HuggingfaceSubject):
        candidate.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                       recording_type=ArtificialSubject.RecordingType.spikerate_exact)
        suite = compute_surprisals(candidate, 'test_suite.json')
        predictions = evaluate(suite)
        targets = [True]
        for x in range(len(predictions)-1):
            targets.append(True)
        metric = load_metric('accuracy')
        score = metric(predictions, targets)
        return score


benchmarks['SG-TSE'] = SyntaxGymTargetedSyntacticEval
