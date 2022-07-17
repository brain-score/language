import logging
# import re
# import string

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_language import load_dataset, load_metric, benchmarks
from brainscore_language.artificial_subject import ArtificialSubject
from lm_zoo import get_registry
from syntaxgym import compute_surprisals, evaluate
from brainscore_language.plugins.sg_targeted_syntactic_evaluation import data

logger = logging.getLogger(__name__)

BIBTEX = """MIT Psycholinguistics lab; SyntaxGym, LM-Zoo; ,
  title={SyntaxGym Targeted Syntactic Evaluations},
  author={MIT Pscyho-linguistics Lab},
  conference={N/A},
  url={https://syntaxgym.org/},
  year={2022}
}"""

class SyntaxGymTargetedSyntacticEval(BenchmarkBase):
    def __init__(self):
        super(SyntaxGymTargetedSyntacticEval, self).__init__(
            identifier="SGTargetedSyntacticEval",
            version=1,
            parent='engineering',
            ceiling_func=None,
            bibtex=BIBTEX)  # TODO: I think this should go into the data plugin somehow (Martin)
            # self.data = load_dataset('SG-TSE') #instead, path to test_suite.json is imported from data.py

    def __call__(self, candidate: ArtificialSubject):
        #  candidate.perform_task(ArtificialSubject.Task.next_word)
        model = get_registry()[candidate.identifier()]
        suite = compute_surprisals(model, data.testsuitepath)   #one of the main SyntaxGym functions
        results = evaluate(suite)                               #one of the main SyntaxGym functions
        score=(sum(results["result"]) / len(results["result"]))
        return score

benchmarks['SG-TSE'] = SyntaxGymTargetedSyntacticEval
