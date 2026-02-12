"""
OASM: Orthogonal Autocorrelated Sequences Model.

A confound baseline model following Hadidi et al. (2025) "Illusions of Alignment Between
Large Language Models and Brains Emerge From Fragile Methods and Overlooked Confounds".

The model constructs an N x N identity matrix (N = total stimuli) and applies Gaussian
smoothing within each block (passage/sentence/story) along axis=1. This captures temporal
autocorrelation -- the fact that brain responses to nearby stimuli are more similar --
without encoding any linguistic content whatsoever.

Reference:
    Hadidi et al. (2025). bioRxiv. https://doi.org/10.1101/2025.03.09.642245
    Code: https://github.com/ebrahimfeghhi/beyond-brainscore
"""

import copy
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Union, List, Dict

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly, merge_data_arrays
from brainscore_language.artificial_subject import ArtificialSubject


class OASMSubject(ArtificialSubject):
    """
    Orthogonal Autocorrelated Sequences Model (OASM) for brain-score evaluation.

    For each block of K stimuli (passage, sentence, or story), constructs a K x K identity
    matrix and applies ``scipy.ndimage.gaussian_filter1d`` along axis=1 with the given sigma.
    Each block is placed at a unique offset in a fixed-size feature space, maintaining
    between-block orthogonality while introducing within-block temporal autocorrelation.

    This is mathematically equivalent to the paper's full N x N construction, adapted for
    brain-score's per-block ``digest_text`` calling convention.

    :param identifier: Unique model identifier (e.g., 'oasm-sigma1.0').
    :param sigma: Gaussian smoothing width. Must be >= 0. sigma=0 gives pure identity.
    :param max_features: Fixed dimensionality of the feature space. Must be >= total
        number of stimuli across all blocks in a benchmark run. Default 2000.
    """

    def __init__(self, identifier: str, sigma: float, max_features: int = 2000):
        self._identifier = identifier
        self._sigma = sigma
        self._max_features = max_features
        self._neural_recordings: list = []
        self._offset: int = 0

    def identifier(self) -> str:
        return self._identifier

    def start_behavioral_task(self, task: ArtificialSubject.Task):
        raise NotImplementedError("OASM encodes no linguistic content and cannot perform behavioral tasks")

    def start_neural_recording(self, recording_target: ArtificialSubject.RecordingTarget,
                               recording_type: ArtificialSubject.RecordingType):
        self._neural_recordings.append((recording_target, recording_type))
        self._offset = 0

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, NeuroidAssembly]:
        assert len(self._neural_recordings) > 0, "Must call start_neural_recording before digest_text"

        if isinstance(text, str):
            text = [text]
        if isinstance(text, np.ndarray):
            text = list(text)

        block_size = len(text)

        if self._offset + block_size > self._max_features:
            raise ValueError(
                f"Cumulative stimulus count ({self._offset + block_size}) exceeds "
                f"max_features ({self._max_features}). Increase max_features."
            )

        # Build block: K x K identity, smoothed along axis=1
        block_features = np.eye(block_size, dtype=np.float64)
        if self._sigma > 0 and block_size > 1:
            block_features = gaussian_filter1d(block_features, sigma=self._sigma, axis=1)

        # Embed into D-dimensional space at the current offset
        features = np.zeros((block_size, self._max_features), dtype=np.float64)
        features[:, self._offset:self._offset + block_size] = block_features
        self._offset += block_size

        # Package each stimulus as a NeuroidAssembly
        output = {'behavior': [], 'neural': []}
        for part_number, stimulus in enumerate(text):
            stimuli_coords = {
                'stimulus': ('presentation', [stimulus]),
                'part_number': ('presentation', [part_number]),
            }
            neural_assembly = self._package_representations(features[part_number], stimuli_coords=stimuli_coords)
            output['neural'].append(neural_assembly)

        output['neural'] = merge_data_arrays(output['neural']).sortby('part_number')
        return output

    def _package_representations(self, representation_values: np.ndarray,
                                 stimuli_coords: dict) -> NeuroidAssembly:
        """Package a feature vector as a NeuroidAssembly matching the brain-score interface."""
        layer_name = f'oasm_sigma{self._sigma}'
        num_units = len(representation_values)

        neuroid_coords = {
            'layer': ('neuroid', [layer_name] * num_units),
            'neuron_number_in_layer': ('neuroid', np.arange(num_units)),
            'neuroid_id': ('neuroid', [f'{layer_name}--{i}' for i in range(num_units)]),
        }

        layer_representations = NeuroidAssembly(
            [representation_values],
            coords={**stimuli_coords, **neuroid_coords},
            dims=['presentation', 'neuroid'])

        representations = []
        for recording_target, recording_type in self._neural_recordings:
            current_representations = copy.deepcopy(layer_representations)
            current_representations['recording_target'] = 'neuroid', [recording_target] * num_units
            current_representations['recording_type'] = 'neuroid', [recording_type] * num_units
            current_representations = type(current_representations)(current_representations)
            representations.append(current_representations)
        representations = merge_data_arrays(representations)
        return representations
