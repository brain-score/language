import xarray as xr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.data.tuckute2024 import BIBTEX

from tqdm import tqdm

def Tuckute2024_linear():
    return _Tuckute2024(metric='linear_pearsonr')

def Tuckute2024_ridge():
    return _Tuckute2024(metric='ridge_pearsonr')

def Tuckute2024_rdm():
    return _Tuckute2024(metric='rdm')

def Tuckute2024_cka():
    return _Tuckute2024(metric='cka')

class _Tuckute2024(BenchmarkBase):

    def __init__(self, metric):
        identifier = f"Tuckute2024-{metric}"
        self.data = load_dataset("Tuckute2024.language")
        self.metric = load_metric(metric)
     
        super(_Tuckute2024, self).__init__(
            identifier=identifier,
            version=1,
            parent='neural_language',
            ceiling=None,
            bibtex=BIBTEX)

    def __call__(self, candidate: ArtificialSubject):
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                         recording_type=ArtificialSubject.RecordingType.fMRI)

        stimuli = self.data['stimulus']
        sentences = self.data['stimulus_id'].values
        predictions = []
        for sentence_id in tqdm(sorted(set(sentences))):  # go over individual stories, sorting to keep consistency across runs
            sentence_indexer = [stimulus_sentence == sentence_id for stimulus_sentence in sentences]
            sentence_stimuli = stimuli[sentence_indexer]
            stimuli_values = sentence_stimuli.values
            sentence_predictions = candidate.digest_text(stimuli_values)["neural"]

            sentence_predictions['stimulus_id'] = 'presentation', sentence_stimuli['stimulus_id'].values
            predictions.append(sentence_predictions)
            
        scores = {}
        predictions = xr.concat(predictions, dim='presentation')
        layer_names = np.unique(predictions['layer'].data)
        layer_names = [layer_names] if isinstance(layer_names, str) else layer_names  # if only one layer, make it a list for consistency
        for layer_name in layer_names:
            raw_score = self.metric(predictions.sel(layer=layer_name), self.data)
            scores[layer_name] = raw_score

        return scores
    