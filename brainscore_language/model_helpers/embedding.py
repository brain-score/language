import copy
import functools
import logging
import numpy as np
import re
from gensim.models.keyedvectors import KeyedVectors
from numpy.core import defchararray
from pathlib import Path
from tqdm import tqdm
from typing import Union, List, Dict, Tuple

from brainio.assemblies import NeuroidAssembly, merge_data_arrays
from brainscore_language.artificial_subject import ArtificialSubject


def mean_over_words(sentence_features):
    sentence_mean = np.mean(sentence_features, axis=0)  # average across words within a sentence
    return sentence_mean


class EmbeddingSubject(ArtificialSubject):
    """
    Lookup-table for word inputs.
    """

    def __init__(self, identifier: str, lookup,
                 layer_name: str = 'projection', average_representations=mean_over_words):
        self._identifier = identifier
        self._logger = logging.getLogger(self.__class__.__name__)
        self._lookup = lookup
        self._layer_name = layer_name
        self._average_representations = average_representations
        self.neural_recordings: List[Tuple] = []  # list of `(recording_target, recording_type)` tuples to record

    def identifier(self):
        return self._identifier

    def start_behavioral_task(self, task: ArtificialSubject.Task):
        raise NotImplementedError("Embedding models do not support behavioral tasks")

    def start_neural_recording(self, recording_target: ArtificialSubject.RecordingTarget,
                               recording_type: ArtificialSubject.RecordingType):
        self.neural_recordings.append((recording_target, recording_type))

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, NeuroidAssembly]:
        assert len(self.neural_recordings) > 0, "Unspecified what to output when not recording"

        if type(text) == str:
            text = [text]

        output = {'behavior': [], 'neural': []}
        text_iterator = tqdm(text, desc='digest text') if len(text) > 100 else text  # show progress bar if many parts
        for part_number, text_part in enumerate(text_iterator):
            representations = self._encode_sentence(text_part)  # encode every word
            representations = self._average_representations(representations)  # reduce to single representation
            # package
            stimuli_coords = {'stimulus': ('presentation', [text_part]),
                              'part_number': ('presentation', [part_number])}
            neural_assembly = self.package_representations(representations, stimuli_coords=stimuli_coords)
            output['neural'].append(neural_assembly)
        output['neural'] = merge_data_arrays(output['neural']).sortby('part_number') if output['neural'] else None
        return output

    def _encode_sentence(self, text_part: str) -> np.ndarray:
        words = text_part.split()
        feature_vectors = []
        for word in words:
            word = remove_punctuation(word)
            word = word.rstrip("'s")
            features = self._lookup[word]
            feature_vectors.append(features)
        return np.array(feature_vectors)

    def package_representations(self, representation_values: np.ndarray, stimuli_coords):
        num_units = representation_values.shape[-1]
        neuroid_coords = {
            'layer': ('neuroid', [self._layer_name] * len(representation_values)),
            'neuron_number_in_layer': ('neuroid', np.arange(num_units)),
        }
        neuroid_coords['neuroid_id'] = 'neuroid', functools.reduce(defchararray.add, [
            neuroid_coords['layer'][1], '--', neuroid_coords['neuron_number_in_layer'][1].astype(str)])
        layer_representations = NeuroidAssembly(
            [representation_values],
            coords={**stimuli_coords, **neuroid_coords},
            dims=['presentation', 'neuroid'])
        # repeat layer representations for every recording
        representations = []
        for recording_target, recording_type in self.neural_recordings:
            current_representations = copy.deepcopy(layer_representations)
            current_representations['recording_target'] = 'neuroid', [recording_target] * num_units
            current_representations['recording_type'] = 'neuroid', [recording_type] * num_units
            current_representations = type(current_representations)(current_representations)  # reindex
            representations.append(current_representations)
        representations = merge_data_arrays(representations)
        return representations


class _GensimLookup(dict):
    def __init__(self, model: KeyedVectors, vector_size: int):
        super(_GensimLookup, self).__init__()
        self._model = model
        self._vector_size = vector_size
        self._logger = logging.getLogger(self.__class__.__name__)

    def __getitem__(self, word):
        try:
            return self._model[word]
        except KeyError:  # not in vocabulary
            self._logger.warning(f"Word {word} not present in model")
            return np.zeros((self._vector_size,))


class GensimKeyedVectorsSubject(EmbeddingSubject):
    """
    Lookup-table models in the gensim library using KeyedVectors.
    """

    def __init__(self, identifier: str, weights_file: Union[str, Path], vector_size: int,
                 weights_file_binary: bool = False, weights_file_no_header: bool = False,
                 layer_name: str = 'projection', average_representations=mean_over_words):
        model = KeyedVectors.load_word2vec_format(weights_file,
                                                  binary=weights_file_binary, no_header=weights_file_no_header)
        lookup = _GensimLookup(model=model, vector_size=vector_size)
        super(GensimKeyedVectorsSubject, self).__init__(identifier=identifier, lookup=lookup, layer_name=layer_name,
                                                        average_representations=average_representations)


def remove_punctuation(word):
    """ Remove dots, question marks, exclamation marks, and commas (`.?!,`) from the word """
    return re.sub(r'[\.\?\!,:]', '', word)
