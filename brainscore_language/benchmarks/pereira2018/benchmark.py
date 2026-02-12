import numpy as np
import xarray as xr

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.data.pereira2018 import BIBTEX
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_language.utils.s3 import load_from_s3
from brainscore_language.benchmarks.blank2014.ceiling import ExtrapolationCeiling


def Pereira2018_243sentences_ridge():
    return _Pereira2018Experiment(experiment='243sentences', metric="ridge_pearsonr",
    crossvalidation_kwargs=dict(
        split_coord="story",
        kfold="group",
        random_state=1234
    )
)

def Pereira2018_384sentences_ridge():
    return _Pereira2018Experiment(experiment='384sentences', metric="ridge_pearsonr",
    crossvalidation_kwargs=dict(
        split_coord="story",
        kfold="group",
        random_state=1234
    )
)


def Pereira2018_243sentences_linear():
    return _Pereira2018Experiment(experiment='243sentences', metric="linear_pearsonr", ceiling_s3_kwargs=dict(
        version_id='CHl_9aFHIWVnPW_njePfy28yzggKuUPw',
        sha1='5e23de899883828f9c886aec304bc5aa0f58f66c',
        raw_kwargs=dict(
            version_id='uZye03ENmn.vKB5mARUGhcIY_DjShtPD',
            sha1='525a6ac8c14ad826c63fdd71faeefb8ba542d5ac',
            raw_kwargs=dict(
                version_id='XVTo58Po5YrNjTuDIWrmfHI0nbN2MVZa',
                sha1='34ba453dc7e8a19aed18cc9bca160e97b4a80be5'
            )
        ),
    ),
    crossvalidation_kwargs=dict(
        split_coord="story",
        kfold="group",
        random_state=1234
    )
)

def Pereira2018_384sentences_linear():
    return _Pereira2018Experiment(experiment='384sentences', metric="linear_pearsonr", ceiling_s3_kwargs=dict(
        version_id='sjlnXr5wXUoGv6exoWu06C4kYI0KpZLk',
        sha1='fc895adc52fd79cea3040961d65d8f736a9d3e29',
        raw_kwargs=dict(
            version_id='Hi74r9UKfpK0h0Bjf5DL.JgflGoaknrA',
            sha1='ce2044a7713426870a44131a99bfc63d8843dae0',
            raw_kwargs=dict(
                version_id='m4dq_ouKWZkYtdyNPMSP0p6rqb7wcYpi',
                sha1='fe9fb24b34fd5602e18e34006ac5ccc7d4c825b8'
            )
        ),
    ),
    crossvalidation_kwargs=dict(
        split_coord="story",
        kfold="group",
        random_state=1234
    )
)


class _Pereira2018Experiment(BenchmarkBase):
    """
    Evaluate model ability to predict neural activity in the human language system in response to natural sentences,
    recorded by Pereira et al. 2018.
    Alignment of neural activity between model and human subjects is evaluated via cross-validated linear predictivity.

    This benchmark builds off the Pereira2018 benchmark introduced
    in Schrimpf et al. 2021 (https://www.pnas.org/doi/10.1073/pnas.2105646118), but:

    * computes neural alignment to each of the two experiments ({243,384}sentences) separately, as well as ceilings
    * requires the model to have committed to neural readouts (e.g. layer 41 corresponds to the language system),
        rather than testing every layer separately

    Each of these benchmarks evaluates one of the two experiments, the overall Pereira2018-linear score is the mean of
    the two ceiling-normalized scores.
    """

    def __init__(self, experiment: str,
            metric: str, 
            ceiling_s3_kwargs: dict = {}, 
            crossvalidation_kwargs: dict = {}, 
            atlas: str = 'language',
        ):
        self.data = self._load_data(experiment, atlas=atlas)
        self.metric = load_metric(metric, crossvalidation_kwargs=crossvalidation_kwargs)
        identifier = f"Pereira2018.{experiment}-{metric.split('_')[0]}"
        if ceiling_s3_kwargs:
            ceiling = self._load_ceiling(identifier=identifier, **ceiling_s3_kwargs)
        else:
            ceiler = ExtrapolationCeiling(subject_column='subject')
            ceiling = ceiler(assembly=self.data, metric=self.metric)

        super(_Pereira2018Experiment, self).__init__(
            identifier=identifier,
            version=1,
            parent='Pereira2018-linear',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def _load_data(self, experiment: str, atlas: str) -> NeuroidAssembly:
        lang_data = load_dataset('Pereira2018.language')
        data = load_dataset(f'Pereira2018.{atlas}')
        data.coords["presentation"] = lang_data.coords["presentation"]
        data = data.sel(experiment=experiment)  # filter experiment
        data = data.dropna('neuroid')  # not all subjects have done both experiments, drop those that haven't
        data.attrs['identifier'] = f"{data.identifier}.{experiment}"
        return data

    def _load_ceiling(self, identifier: str, version_id: str, sha1: str, assembly_prefix="ceiling_", raw_kwargs=None):
        ceiling = load_from_s3(identifier, cls=Score, assembly_prefix=assembly_prefix, version_id=version_id, sha1=sha1)
        if raw_kwargs:  # recursively load raw attributes
            raw = self._load_ceiling(identifier=identifier, assembly_prefix=assembly_prefix + "raw_", **raw_kwargs)
            ceiling.attrs['raw'] = raw
        return ceiling

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                         recording_type=ArtificialSubject.RecordingType.fMRI)
        stimuli = self.data['stimulus']
        passages = self.data['passage_label'].values
        predictions = []
        for passage in sorted(set(passages)):  # go over individual passages, sorting to keep consistency across runs
            passage_indexer = [stimulus_passage == passage for stimulus_passage in passages]
            passage_stimuli = stimuli[passage_indexer]
            passage_predictions = candidate.digest_text(passage_stimuli.values)['neural']
            passage_predictions['stimulus_id'] = 'presentation', passage_stimuli['stimulus_id'].values
            try:
                passage_predictions['passage_index']
            except KeyError:
                passage_predictions['passage_index'] = 'presentation', passage_stimuli['passage_index'].values
            try:
                passage_predictions['story']
            except KeyError:
                passage_predictions['story'] = 'presentation', passage_stimuli['story'].values
            predictions.append(passage_predictions)
    
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
