import logging

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric, benchmarks
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_language.utils.s3 import load_from_s3
from .data import BIBTEX

logger = logging.getLogger(__name__)


def Pereira2018Experiment243():
    return _Pereira2018ExperimentLinear(experiment='243sentences', ceiling_s3_kwargs=dict(
        version_id='yJ2mOYGaBM9wNy3A7lD29N62j7LW.qzJ',
        sha1='5e23de899883828f9c886aec304bc5aa0f58f66c'
    ))


def Pereira2018Experiment384():
    return _Pereira2018ExperimentLinear(experiment='384sentences', ceiling_s3_kwargs=dict(
        version_id='YJGsV8d1Vsvluz6JL6xuygOh_uyw3f8A',
        sha1='fc895adc52fd79cea3040961d65d8f736a9d3e29'
    ))


class _Pereira2018ExperimentLinear(BenchmarkBase):
    """
    Evaluate model ability to predict neural activity in the human language system in response to natural sentences,
    recorded by Pereira et al. 2018.
    Alignment of neural activity between model and human subjects is evaluated via cross-validated linear predictivity.

    This benchmark builds off the behavioral benchmark introduced
    in Schrimpf et al. 2021 (https://www.pnas.org/doi/10.1073/pnas.2105646118), but:

    * computes neural alignment to each of the two experiments ({243,384}sentences) separately, as well as ceilings
    * requires the model to have committed to neural readouts (e.g. layer 41 corresponds to the language system),
        rather than testing every layer separately

    Each of these benchmarks evaluates one of the two experiments, the overall Pereira2018-linear score is the mean of
    the two ceiling-normalized scores.
    """

    def __init__(self, experiment, ceiling_s3_kwargs):
        self.data = self.load_data(experiment)
        self.metric = load_metric('linear_pearsonr')
        identifier = f'Pereira2018.{experiment}-linear'
        ceiling = load_from_s3(identifier, cls=Score, assembly_prefix="ceiling_", **ceiling_s3_kwargs)
        super(_Pereira2018ExperimentLinear, self).__init__(
            identifier=identifier,
            version=1,
            parent='Pereira2018-linear',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def load_data(self, experiment):
        data = load_dataset('Pereira2018.language')
        data = data.sel(experiment=experiment)  # filter experiment
        data = data.dropna('neuroid')  # not all subjects have done both experiments, drop those that haven't
        data.attrs['identifier'] = f"{data.identifier}.{experiment}"
        return data

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                           recording_type=ArtificialSubject.RecordingType.fMRI)
        stimuli = self.data['stimulus']
        predictions = candidate.digest_text(stimuli.values)['neural']
        predictions['stimulus_id'] = 'presentation', stimuli['stimulus_id'].values
        raw_score = self.metric(predictions, self.data)
        score = ceiling_normalize(raw_score, self.ceiling)
        return score


benchmarks['Pereira2018.243-linear'] = Pereira2018Experiment243
benchmarks['Pereira2018.384-linear'] = Pereira2018Experiment384
