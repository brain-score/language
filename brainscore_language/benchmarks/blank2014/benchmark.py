import numpy as np
import xarray as xr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.benchmarks.blank2014.ceiling import ExtrapolationCeiling
from brainscore_language.data.blank2014 import BIBTEX
from brainscore_language.utils.ceiling import ceiling_normalize


def Blank2014_ridge():
    return Blank2014(metric="ridge_pearsonr", 
    cross_validation_kwargs=dict(
        splits=8,
        split_coord="story",
        kfold="group",
        random_state=1234
    )
)

def Blank2014_linear():
    return Blank2014(metric="linear_pearsonr", 
    cross_validation_kwargs=dict(
        splits=8,
        split_coord="story",
        kfold="group",
        random_state=1234
    )
)

class Blank2014(BenchmarkBase):
    """
    Evaluate model ability to predict neural activity in human language system functional regions of interest (fROIs)
    in response to natural stories, recorded by Blank et al. 2014.
    Alignment of neural activity between model and human subjects is evaluated via cross-validated linear predictivity.

    This benchmark builds off the Blank2014 benchmark introduced in Schrimpf et al. 2021
    (https://www.pnas.org/doi/10.1073/pnas.2105646118), but requires the model to have committed to neural readouts
    (e.g. "layer 41 corresponds to the language system"), rather than testing every layer separately.
    """

    def __init__(self, metric: str, cross_validation_kwargs=None):
        self.data = load_dataset('Blank2014.fROI')
        self.metric = load_metric(metric, crossvalidation_kwargs=cross_validation_kwargs)
        ceiler = ExtrapolationCeiling()
        ceiling = ceiler(assembly=self.data, metric=self.metric)
        super(Blank2014, self).__init__(
            identifier=f'Blank2014-{metric}',
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
            try:
                story_predictions['story']
            except KeyError:
                story_predictions['story'] = 'presentation', story_stimuli['story'].values
            predictions.append(story_predictions)

        predictions = xr.concat(predictions, dim='presentation')
        layer_names = np.unique(predictions['layer'].data)
        layer_names = [layer_names] if isinstance(layer_names, str) else layer_names
        layer_scores = {}
        for layer_name in layer_names:
            raw_score = self.metric(predictions.sel(layer=layer_name), self.data)
            layer_scores[layer_name] = ceiling_normalize(raw_score, self.ceiling)

        score = Score(np.mean(list(layer_scores.values())))
        score.attrs['layer_scores'] = layer_scores
        return score
