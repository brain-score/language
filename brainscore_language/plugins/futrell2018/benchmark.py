import logging

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric, benchmarks
from brainscore_language.artificial_subject import ArtificialSubject

logger = logging.getLogger(__name__)

BIBTEX = """"""  # todo: who gets credit here? data/benchmark/both?


class Futrell2018Pearsonr(BenchmarkBase):
    """
    Evaluate model ability to predict reading times on the natural stories corpus introduced in Futrell et al. 2018.
    Alignment of reading times between model and human subjects is evaluated via Pearson correlation.

    This benchmark builds off the behavioral benchmark introduced in Schrimpf et al. 2021, but does not allow for any
    fitting; rather model candidates have to directly output reading times.
    """

    def __init__(self):
        super(Futrell2018Pearsonr, self).__init__(
            identifier='Futrell2018-pearsonr',
            version=1,
            parent='behavior',
            ceiling_func=None,  # todo
            bibtex=BIBTEX)  # TODO: I think this should go into the data plugin somehow
        self.data = load_dataset('Futrell2018')
        self.metric = load_metric('pearsonr')

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.perform_behavioral_task(ArtificialSubject.Task.reading_times)
        stimuli = self.data['word'].values
        predictions = candidate.digest_text(stimuli)['behavior'].values
        targets = self.data.mean('subject')  # compare to "average human"
        score = self.metric(predictions, targets)
        # todo: ceiling normalize
        return score


benchmarks['Futrell2018-pearsonr'] = Futrell2018Pearsonr
