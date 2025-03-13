import xarray as xr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.benchmarks.blank2014.ceiling import ExtrapolationCeiling
from brainscore_language.data.fedorenko2016 import BIBTEX
from brainscore_language.utils.ceiling import ceiling_normalize

from tqdm import tqdm

def Fedorenko2016_linear():
    return Fedorenko2016(metric="linear_pearsonr")

def Fedorenko2016_ridge():
    return Fedorenko2016(metric="ridge_pearsonr")

class Fedorenko2016(BenchmarkBase):

    def __init__(self, metric: str):
        self.data = load_dataset('Fedorenko2016.language')
        
        identifier = f"Fedorenko2016-{metric}"
        self.metric = load_metric(metric)

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
            predictions.append(sentence_predictions)
            
        predictions = xr.concat(predictions, dim='presentation')

        raw_score = self.metric(predictions, self.data)
        scores = ceiling_normalize(raw_score, self.ceiling)

        return scores