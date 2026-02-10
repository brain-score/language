"""
OASM: Ordinal position, Average word length, Sentence length, Mean features.

A simple confound baseline model following Hadidi et al. (2025) "Illusions of Alignment Between
Large Language Models and Brains Emerge From Fragile Methods and Overlooked Confounds".

The paper argues that simple confounding variables — particularly positional signals (ordinal sentence
position within a passage) and word rate (number of words per sentence) — perform competitively with
trained LLMs in predicting brain activity.

This model implements these confound features as an ArtificialSubject for evaluation on Brain-Score:
  - O: Ordinal position of the sentence within the passage
  - A: Average word length (mean characters per word)
  - S: Sentence length (number of words, i.e. word rate)
  - M: Mean character count and other summary statistics
"""

import copy
import numpy as np
from typing import Union, List, Dict

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly, merge_data_arrays
from brainscore_language.artificial_subject import ArtificialSubject


class OASMSubject(ArtificialSubject):
    """
    Confound baseline model that uses only simple sentence-level statistics as "neural" representations.

    For each sentence, the feature vector consists of:
      - Ordinal position (0-indexed position of sentence in the passage)
      - Normalized ordinal position (position / total sentences in passage)
      - Number of words (word rate)
      - Log number of words
      - Average word length in characters
      - Total character count
      - Log character count
      - Number of unique words
      - Max word length
      - Min word length

    These features capture the confounding variables identified by Hadidi et al. (2025):
    positional signals and word rate.
    """

    def __init__(self, identifier: str = 'oasm'):
        self._identifier = identifier
        self.neural_recordings: list = []

    def identifier(self) -> str:
        return self._identifier

    def start_behavioral_task(self, task: ArtificialSubject.Task):
        raise NotImplementedError("OASM confound model does not support behavioral tasks")

    def start_neural_recording(self, recording_target: ArtificialSubject.RecordingTarget,
                               recording_type: ArtificialSubject.RecordingType):
        self.neural_recordings.append((recording_target, recording_type))

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, NeuroidAssembly]:
        assert len(self.neural_recordings) > 0, "Must call start_neural_recording before digest_text"

        if isinstance(text, str):
            text = [text]
        if isinstance(text, np.ndarray):
            text = list(text)

        num_sentences = len(text)
        output = {'behavior': [], 'neural': []}

        for part_number, sentence in enumerate(text):
            features = self._compute_features(sentence, ordinal_position=part_number,
                                              total_sentences=num_sentences)
            stimuli_coords = {
                'stimulus': ('presentation', [sentence]),
                'part_number': ('presentation', [part_number]),
            }
            neural_assembly = self._package_representations(features, stimuli_coords=stimuli_coords)
            output['neural'].append(neural_assembly)

        output['neural'] = merge_data_arrays(output['neural']).sortby('part_number') if output['neural'] else None
        return output

    def _compute_features(self, sentence: str, ordinal_position: int, total_sentences: int) -> np.ndarray:
        """
        Compute simple confound features for a single sentence.

        Features:
          0: Ordinal position (0-indexed)
          1: Normalized ordinal position (position / total)
          2: Number of words (word rate)
          3: Log(1 + number of words)
          4: Average word length in characters
          5: Total character count (excluding spaces)
          6: Log(1 + total character count)
          7: Number of unique words
          8: Max word length
          9: Min word length
        """
        words = sentence.split()
        num_words = len(words) if words else 1
        word_lengths = [len(w) for w in words] if words else [0]

        features = np.array([
            ordinal_position,                                               # O: ordinal position
            ordinal_position / max(total_sentences - 1, 1),                 # O: normalized position
            num_words,                                                       # S: sentence length (word rate)
            np.log1p(num_words),                                            # S: log word count
            np.mean(word_lengths),                                          # A: average word length
            sum(word_lengths),                                              # M: total char count (no spaces)
            np.log1p(sum(word_lengths)),                                    # M: log char count
            len(set(w.lower() for w in words)) if words else 0,            # M: unique word count
            max(word_lengths),                                              # M: max word length
            min(word_lengths),                                              # M: min word length
        ], dtype=np.float64)

        return features

    def _package_representations(self, representation_values: np.ndarray, stimuli_coords: dict) -> NeuroidAssembly:
        """Package feature vector as a NeuroidAssembly matching the Brain-Score interface."""
        layer_name = 'oasm_features'
        num_units = len(representation_values)
        feature_names = [
            'ordinal_position', 'normalized_position',
            'word_count', 'log_word_count',
            'avg_word_length',
            'total_char_count', 'log_char_count',
            'unique_word_count', 'max_word_length', 'min_word_length',
        ]

        neuroid_coords = {
            'layer': ('neuroid', [layer_name] * num_units),
            'neuron_number_in_layer': ('neuroid', np.arange(num_units)),
            'feature_name': ('neuroid', feature_names),
        }
        neuroid_coords['neuroid_id'] = 'neuroid', [f'{layer_name}--{i}' for i in range(num_units)]

        layer_representations = NeuroidAssembly(
            [representation_values],
            coords={**stimuli_coords, **neuroid_coords},
            dims=['presentation', 'neuroid'])

        # Repeat layer representations for every recording
        representations = []
        for recording_target, recording_type in self.neural_recordings:
            current_representations = copy.deepcopy(layer_representations)
            current_representations['recording_target'] = 'neuroid', [recording_target] * num_units
            current_representations['recording_type'] = 'neuroid', [recording_type] * num_units
            current_representations = type(current_representations)(current_representations)  # reindex
            representations.append(current_representations)
        representations = merge_data_arrays(representations)
        return representations
