"""Unified-interface variant of Pereira2018 benchmark.

Calls `model.process(stimulus_set)` directly instead of legacy
`model.digest_text(text_array)`. Uses unified `start_recording('language_system')`
instead of `start_neural_recording(...)`. Scores must match the legacy
variant.
"""

import pandas as pd
import xarray as xr

from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_language.utils.ceiling import ceiling_normalize
from .benchmark import _Pereira2018ExperimentLinear


class _Pereira2018ExperimentLinearUnified(_Pereira2018ExperimentLinear):
    """Pereira2018 benchmark that calls process() directly.

    Differences from the parent:
    - Uses `start_recording('language_system', recording_type='fMRI')` instead
      of `start_neural_recording(...)`. BrainScoreModel's compat method
      delegates to start_recording, but new models can implement it natively.
    - Constructs a StimulusSet per passage and calls `process(stimulus_set)`
      instead of `digest_text(text_array)['neural']`. The model returns a
      NeuroidAssembly directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._identifier = self.identifier + '-unified'

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, value):
        self._identifier = value

    def __call__(self, candidate) -> Score:
        # Unified: start_recording instead of start_neural_recording
        candidate.start_recording('language_system', recording_type='fMRI')

        stimuli = self.data['stimulus']
        passages = self.data['passage_label'].values
        predictions = []
        for passage in sorted(set(passages)):
            passage_indexer = [stimulus_passage == passage for stimulus_passage in passages]
            passage_stimuli = stimuli[passage_indexer]
            sentences = list(passage_stimuli.values)
            stimulus_ids = list(passage_stimuli['stimulus_id'].values)

            # Build a proper StimulusSet for process()
            passage_stimulus_set = StimulusSet(pd.DataFrame({
                'sentence': sentences,
                'stimulus_id': stimulus_ids,
            }))
            passage_stimulus_set.identifier = (
                f'pereira_passage_{passage}'
            )

            # Unified: process() instead of digest_text(...)['neural']
            passage_predictions = candidate.process(passage_stimulus_set)
            passage_predictions['stimulus_id'] = (
                'presentation', stimulus_ids,
            )
            predictions.append(passage_predictions)

        predictions = xr.concat(predictions, dim='presentation')
        raw_score = self.metric(predictions, self.data)
        score = ceiling_normalize(raw_score, self.ceiling)
        return score


def Pereira2018_243sentences_unified():
    return _Pereira2018ExperimentLinearUnified(
        experiment='243sentences',
        ceiling_s3_kwargs=dict(
            sha1='5e23de899883828f9c886aec304bc5aa0f58f66c',
            raw_kwargs=dict(
                sha1='525a6ac8c14ad826c63fdd71faeefb8ba542d5ac',
                raw_kwargs=dict(
                    sha1='34ba453dc7e8a19aed18cc9bca160e97b4a80be5'
                )
            )
        ),
    )


def Pereira2018_384sentences_unified():
    return _Pereira2018ExperimentLinearUnified(
        experiment='384sentences',
        ceiling_s3_kwargs=dict(
            sha1='fc895adc52fd79cea3040961d65d8f736a9d3e29',
            raw_kwargs=dict(
                sha1='ce2044a7713426870a44131a99bfc63d8843dae0',
                raw_kwargs=dict(
                    sha1='fe9fb24b34fd5602e18e34006ac5ccc7d4c825b8'
                )
            )
        ),
    )
