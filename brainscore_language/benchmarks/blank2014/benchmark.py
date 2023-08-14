import xarray as xr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.benchmarks.blank2014.ceiling import ExtrapolationCeiling
from brainscore_language.data.blank2014 import BIBTEX
from brainscore_language.utils.ceiling import ceiling_normalize


class Blank2014Linear(BenchmarkBase):
    """
    Evaluate model ability to predict neural activity in human language system functional regions of interest (fROIs)
    in response to natural stories, recorded by Blank et al. 2014.
    Alignment of neural activity between model and human subjects is evaluated via cross-validated linear predictivity.

    This benchmark builds off the Blank2014 benchmark introduced in Schrimpf et al. 2021
    (https://www.pnas.org/doi/10.1073/pnas.2105646118), but requires the model to have committed to neural readouts
    (e.g. "layer 41 corresponds to the language system"), rather than testing every layer separately.
    """

    def __init__(self):
        self.data = load_dataset('Blank2014.fROI')
        self.metric = load_metric('linear_pearsonr')
        ceiler = ExtrapolationCeiling()
        ceiling = ceiler(assembly=self.data, metric=self.metric)
        super(Blank2014Linear, self).__init__(
            identifier='Blank2014-linear',
            version=1,
            parent='neural_language',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                         recording_type=ArtificialSubject.RecordingType.fMRI)
        stimuli = self.data['stimulus']
        stories = self.data['story'].values
        predictions = []
        for story in sorted(set(stories)):  # go over individual stories, sorting to keep consistency across runs
            story_indexer = [stimulus_story == story for stimulus_story in stories]
            story_stimuli = stimuli[story_indexer]
            story_predictions = candidate.digest_text(story_stimuli.values)['neural']
            story_predictions['stimulus_id'] = 'presentation', story_stimuli['stimulus_id'].values
            predictions.append(story_predictions)
        predictions = xr.concat(predictions, dim='presentation')
        raw_score = self.metric(predictions, self.data)
        score = ceiling_normalize(raw_score, self.ceiling)
        return score
