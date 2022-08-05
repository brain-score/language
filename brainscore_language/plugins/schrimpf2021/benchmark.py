import logging

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric, benchmarks
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.plugins.schrimpf2021.data import BIBTEX_PEREIRA2018

logger = logging.getLogger(__name__)


class Pereira2018Linear(BenchmarkBase):
    """
    Evaluate model ability to predict neural activity in the human language system in response to natural sentences,
    recorded by Pereira et al. 2018.
    Alignment of neural activity between model and human subjects is evaluated via cross-validated linear predictivity.
    This benchmark was first introduced in Schrimpf et al. 2021.
    """

    def __init__(self):
        self.data = load_dataset('Pereira2018.language_system')
        self.metric = load_metric('linear_predictivity')
        ceiler = None
        super(Pereira2018Linear, self).__init__(
            identifier='Pereira2018-linear',
            version=1,
            parent='neural',
            ceiling_func=lambda: ceiler(self.data),
            bibtex=BIBTEX_PEREIRA2018)

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                           recording_type=ArtificialSubject.RecordingType.spikerate_exact)
        stimuli = self.data  # todo
        predictions = candidate.digest_text(stimuli)['neural']
        raw_score = self.metric(predictions, self.data)
        return raw_score
        # score = ceiling_normalize(raw_score, self.ceiling) # todo
        # return score


benchmarks['Pereira2018-linear'] = Pereira2018Linear
