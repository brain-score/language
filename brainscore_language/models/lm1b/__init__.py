import functools
import logging
from pathlib import Path
import urllib.request
from typing import Tuple, Union, List, Dict

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv(
    "TF_CPP_MIN_LOG_LEVEL", "3"
)  # Disable verbose TF C++ output

import numpy as np
import tensorflow.compat.v1 as tf
import xarray as xr
from numpy.core import defchararray
from tqdm import tqdm
from google.protobuf import text_format

from brainio.assemblies import DataAssembly, NeuroidAssembly, BehavioralAssembly
from brainscore_language import model_registry, ArtificialSubject
from brainscore_language.utils import fullname

from .data_utils import CharsVocabulary

MAX_WORD_LEN = 50
RESOURCES = [
    "http://download.tensorflow.org/models/LM_LSTM_CNN/graph-2016-09-10.pbtxt",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-base",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-char-embedding",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-lstm",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax0",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax1",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax2",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax3",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax4",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax5",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax6",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax7",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax8",
    "http://download.tensorflow.org/models/LM_LSTM_CNN/vocab-2016-09-10.txt",
]


class LM1B(ArtificialSubject):
    def __init__(
        self,
        model_id: str,
        region_layer_mapping: dict,
    ):
        """
        :param model_id: the model id i.e. name
        """
        self._logger = logging.getLogger(fullname(self))
        self.model_id = model_id
        self.region_layer_mapping = region_layer_mapping

        self.sess, self.encoder_t, self.vocab = self._load_encoder()
        self.ctx_word_ids = None  # keeps track of tokens in context
        self.neural_recordings: List[
            Tuple
        ] = []  # list of `(recording_target, recording_type)` tuples to record

        self.behavioral_task: Union[None, ArtificialSubject.Task] = None
        self.task_function_mapping_dict = {
            ArtificialSubject.Task.next_word: self.predict_next_word,
            ArtificialSubject.Task.reading_times: self.estimate_reading_times,
        }

    def identifier(self):
        return self.model_id

    def start_behavioral_task(self, task: ArtificialSubject.Task):
        self.behavioral_task = task
        self.output_to_behavior = self.task_function_mapping_dict[task]

    def start_neural_recording(
        self,
        recording_target: ArtificialSubject.RecordingTarget,
        recording_type: ArtificialSubject.RecordingType,
    ):
        self.neural_recordings.append((recording_target, recording_type))

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, DataAssembly]:
        """
        :param text: the text to be used for inference e.g. "the quick brown fox"
        :return: assembly of either behavioral output or internal neural representations
        """

        # Convert str text into list format
        if isinstance(text, str):
            text = [text]

        # Prefix first text token with <S>
        sentence_start_token = "<S> "
        text[0] = sentence_start_token + text[0]

        # Override targets and weights state
        targets = np.zeros((1, 1), np.int32)
        weights = np.ones((1, 1), np.float32)

        inputs = np.zeros((1, 1), np.int32)
        char_ids_inputs = np.zeros([1, 1, self.vocab.max_word_length], np.int32)

        # split textual input into word and character ids
        text_parts = []
        for part in text:
            word_ids, char_ids = [], []
            for w in part.split():
                word_ids.append(self.vocab.word_to_id(w))
                char_ids.append(self.vocab.word_to_char_ids(w))
            text_parts.append((word_ids, char_ids))

        # Run model for each context and record behavioral and neural data after each ctx
        output = {"behavior": [], "neural": []}
        text_iterator = (
            tqdm(text_parts, desc="digest text") if len(text_parts) > 1 else text_parts
        )  # show progress bar if multiple parts
        logits = []
        for part_number, text_part in enumerate(text_iterator):
            word_ids, char_ids = text_part
            context = " ".join(text[: part_number + 1])[len(sentence_start_token) :]
            self.ctx_word_ids = np.asarray(word_ids)
            for i in range(len(word_ids)):
                inputs[0, 0] = word_ids[i]
                char_ids_inputs[0, 0, :] = char_ids[i]

                fetches = (
                    self.encoder_t["softmax_out"],
                    *(
                        self.encoder_t[
                            "lstm/%s/control_dependency"
                            % self.region_layer_mapping[recording_target]
                        ]
                        for recording_target, _ in self.neural_recordings
                    ),
                )
                sess_out = self.sess.run(
                    fetches,
                    feed_dict={
                        self.encoder_t["char_inputs_in"]: char_ids_inputs,
                        self.encoder_t["inputs_in"]: inputs,
                        self.encoder_t["targets_in"]: targets,
                        self.encoder_t["target_weights_in"]: weights,
                    },
                )
                softmax, layer_representations = sess_out[0], sess_out[1:]
                logits.append(softmax)

            # format output
            stimuli_coords = {
                "stimulus": ("presentation", [text[part_number]]),
                "context": ("presentation", [context]),
                "part_number": ("presentation", [part_number]),
            }

            if self.behavioral_task:
                # logits (seq_len, batch_size, vocab_size) --> (seq_len, vocab_size) because bz is 1
                logits_squeezed = np.asarray(logits)[:, 0, :]

                # format behavioral output into assembly
                behavioral_output = self.output_to_behavior(logits_squeezed)
                behavior = BehavioralAssembly(
                    [behavioral_output], coords=stimuli_coords, dims=["presentation"]
                )
                output["behavior"].append(behavior)

            if self.neural_recordings:
                representations = self.output_to_representations(
                    layer_representations, stimuli_coords=stimuli_coords
                )
                output["neural"].append(representations)

        # merge over text parts
        self._logger.debug("Merging outputs")
        output["behavior"] = (
            xr.concat(output["behavior"], dim="presentation").sortby("part_number")
            if output["behavior"]
            else None
        )
        output["neural"] = (
            xr.concat(output["neural"], dim="presentation").sortby("part_number")
            if output["neural"]
            else None
        )
        return output

    def estimate_reading_times(self, logits):
        """
        :param logits: the neural network's softmax output (seq_len, vocab_size)
        :return: surprisal (in bits) as a proxy for reading times, following Smith & Levy 2013
            (https://www.sciencedirect.com/science/article/pii/S0010027713000413)
        """

        # get expectation (logits), shifted left by 1
        predicted_logits = logits[-self.ctx_word_ids.shape[0] - 1 : -1]
        actual_tokens = self._vector_to_one_hot(self.ctx_word_ids, logits.shape[1])
        if (
            actual_tokens.shape[0] == predicted_logits.shape[0] + 1
        ):  # multiple tokens for first model input
            actual_tokens = actual_tokens[
                1:
            ]  # we have no prior context to predict the 0th token

        # assume that reading time is additive, i.e. reading time of multiple tokens is
        # the sum of the surprisals of each individual token.
        surprisal = -np.sum(actual_tokens * np.log2(predicted_logits))
        return surprisal.item()

    def predict_next_word(self, logits):
        """
        :param logits: the neural network's softmax output (seq_len, vocab_size)
        :return: predicted next word
        """

        logits[:, 2] = 0  # mask <UNK> token for next word predictions
        pred_id = np.argmax(logits, axis=1)

        # Note that this is currently only predicting the next *token* which might not always be entire words.
        last_model_token_inference = pred_id[-1]
        next_word = self.vocab.id_to_word(last_model_token_inference)

        # `next_word` often includes a space ` ` in front of the actual word. Since the task already tells us to output
        # a word, we can strip the space.
        next_word = next_word.strip()
        return next_word

    def output_to_representations(self, layer_representations, stimuli_coords):
        """Format layer representations as a neuroid assembly"""

        # Re-format layer_representations to contain region, recording type, and layer
        layer_reps_mapping = {}
        for layer_i, representations in enumerate(layer_representations):
            recording_target, recording_type = self.neural_recordings[layer_i]
            layer = self.region_layer_mapping[recording_target]

            key = (recording_target, recording_type, layer)
            layer_reps_mapping[key] = representations
        layer_representations = layer_reps_mapping

        representation_values = np.concatenate(
            list(layer_representations.values()), axis=-1
        )  # concatenate along neuron axis
        neuroid_coords = {
            "layer": (
                "neuroid",
                np.concatenate(
                    [
                        [layer] * values.shape[-1]
                        for (
                            recording_target,
                            recording_type,
                            layer,
                        ), values in layer_representations.items()
                    ]
                ),
            ),
            "region": (
                "neuroid",
                np.concatenate(
                    [
                        [recording_target] * values.shape[-1]
                        for (
                            recording_target,
                            recording_type,
                            layer,
                        ), values in layer_representations.items()
                    ]
                ),
            ),
            "recording_type": (
                "neuroid",
                np.concatenate(
                    [
                        [recording_type] * values.shape[-1]
                        for (
                            recording_target,
                            recording_type,
                            layer,
                        ), values in layer_representations.items()
                    ]
                ),
            ),
            "neuron_number_in_layer": (
                "neuroid",
                np.concatenate(
                    [
                        np.arange(values.shape[-1])
                        for values in layer_representations.values()
                    ]
                ),
            ),
        }
        neuroid_coords["neuroid_id"] = "neuroid", functools.reduce(
            defchararray.add,
            [
                neuroid_coords["layer"][1],
                "--",
                neuroid_coords["neuron_number_in_layer"][1].astype(str),
            ],
        )

        representations = NeuroidAssembly(
            representation_values,
            coords={**stimuli_coords, **neuroid_coords},
            dims=["presentation", "neuroid"],
        )
        return representations

    def _load_encoder(self):
        """Download model resources, if not available, and build model from checkpoints"""

        # Check if model resources are available, otherwise download them
        cur_dirname = Path(__file__).parent
        resources_dir = cur_dirname / "resources"
        if not resources_dir.is_dir():
            self._download_resources(resources_dir)

        gd_file = str(resources_dir / "graph-2016-09-10.pbtxt")
        ckpt_file = str(resources_dir / "ckpt-*")
        vocab_file = str(resources_dir / "vocab-2016-09-10.txt")

        vocab = CharsVocabulary(vocab_file, MAX_WORD_LEN)

        # Start TF session from imported the graph and checkpoints
        with tf.Graph().as_default():
            with tf.gfile.GFile(gd_file, "r") as f:
                s = f.read()
                gd = tf.GraphDef()
                text_format.Merge(s, gd)

            t = {}
            [
                t["states_init"],
                t["lstm/lstm_0/control_dependency"],
                t["lstm/lstm_1/control_dependency"],
                t["softmax_out"],
                t["class_ids_out"],
                t["class_weights_out"],
                t["log_perplexity_out"],
                t["inputs_in"],
                t["targets_in"],
                t["target_weights_in"],
                t["char_inputs_in"],
                t["all_embs"],
                t["softmax_weights"],
                t["global_step"],
            ] = tf.import_graph_def(
                gd,
                {},
                [
                    "states_init",
                    "lstm/lstm_0/control_dependency:0",
                    "lstm/lstm_1/control_dependency:0",
                    "softmax_out:0",
                    "class_ids_out:0",
                    "class_weights_out:0",
                    "log_perplexity_out:0",
                    "inputs_in:0",
                    "targets_in:0",
                    "target_weights_in:0",
                    "char_inputs_in:0",
                    "all_embs_out:0",
                    "Reshape_3:0",
                    "global_step:0",
                ],
                name="",
            )

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            sess.run("save/restore_all", {"save/Const:0": ckpt_file})
            sess.run(t["states_init"])

        return sess, t, vocab

    def _download_resources(self, target_directory: Path):
        """Download pretrained LM1B checkpoint and vocabulary"""

        target_directory.mkdir()
        downloading_iter = tqdm(RESOURCES, desc="Downloading resources for LM1B.")
        for resource in downloading_iter:
            file_name = resource.split("/")[-1]
            file_path = target_directory / file_name
            urllib.request.urlretrieve(resource, file_path)

    def _vector_to_one_hot(self, v, vocab_size):
        """
        Given a vector of indices and a vocab size, returns a one-hot representation of the indices
        as vocab_size-long vectors
        """

        one_hot = np.zeros((len(v), vocab_size))
        one_hot[np.arange(len(v)), v] = 1
        return one_hot


model_registry["lm1b"] = lambda: LM1B(
    model_id="lm1b",
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: "lstm_1"},
)
