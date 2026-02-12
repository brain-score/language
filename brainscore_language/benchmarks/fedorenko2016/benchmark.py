import numpy as np
import xarray as xr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.benchmarks.blank2014.ceiling import ExtrapolationCeiling
from brainscore_language.data.fedorenko2016 import BIBTEX
from brainscore_language.utils.ceiling import ceiling_normalize

from tqdm import tqdm


def Fedorenko2016_ridge():
    return Fedorenko2016(metric="ridge_pearsonr", 
    cross_validation_kwargs=dict(
        split_coord="sentence_id",
        kfold="group",
        random_state=1234
    )
)

def Fedorenko2016_linear():
    return Fedorenko2016(metric="linear_pearsonr")

class Fedorenko2016(BenchmarkBase):

    def __init__(self, metric: str, cross_validation_kwargs=None):
        self.data = load_dataset('Fedorenko2016.language')
        
        identifier = f"Fedorenko2016-{metric}"
        self.metric = load_metric(metric, crossvalidation_kwargs=cross_validation_kwargs)

        ceiler = ExtrapolationCeiling(subject_column="subject_UID")
        ceiling = ceiler(assembly=self.data, metric=self.metric)
         
        super(Fedorenko2016, self).__init__(
            identifier=identifier,
            version=3,
            parent='neural_language',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def __call__(self, candidate: ArtificialSubject):
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                         recording_type=ArtificialSubject.RecordingType.ECoG)

        stimuli = self.data['stimulus']
        sentences = self.data['sentence_id'].values
        predictions = []
        for sentence_id in tqdm(sorted(set(sentences))):  # go over individual stories, sorting to keep consistency across runs
            sentence_indexer = [stimulus_sentence == sentence_id for stimulus_sentence in sentences]
            sentence_stimuli = stimuli[sentence_indexer]
            stimuli_values = sentence_stimuli.values
            sentence_predictions = candidate.digest_text(stimuli_values)["neural"]
            sentence_predictions['stimulus_id'] = 'presentation', sentence_stimuli['stimulus_id'].values
            try:
                sentence_predictions['sentence_id']
            except KeyError:
                sentence_predictions['sentence_id'] = 'presentation', sentence_stimuli['sentence_id'].values
            predictions.append(sentence_predictions)
            
        predictions = xr.concat(predictions, dim='presentation')
        layer_names = np.unique(predictions['layer'].data)
        layer_names = [layer_names] if isinstance(layer_names, str) else layer_names
        layer_scores = {}
        for layer_name in layer_names:
            raw_score = self.metric(predictions.sel(layer=layer_name), self.data)
            layer_scores[layer_name] = ceiling_normalize(raw_score, self.ceiling)

        score = Score(np.mean(list(layer_scores.values())))
        score.attrs['layer_scores'] = layer_scores
        score.attrs['raw'] = Score(np.mean([s.attrs['raw'] for s in layer_scores.values()]))
        score.attrs['ceiling'] = self.ceiling
        return score