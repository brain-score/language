from collections import OrderedDict

import functools
import json
import logging
import multiprocessing
import numpy as np
import re
import subprocess
import sys
import torch
import xarray as xr
from joblib import Parallel, delayed, parallel_backend
from numpy.core import defchararray
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Union, Callable

from brainio.assemblies import DataAssembly, NeuroidAssembly, BehavioralAssembly
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils import fullname


class ContainerSubject(ArtificialSubject):
    """
    Evaluation interface for arbitary containerized models.
    User must install either 'Singularity' or 'Docker' to evaluate container models.

    To add new model, build a container with an entry point that supports the following interface:

    $CONTAINER_BACKEND run $CONTAINER_NAME $ENTRYPOINT \
    --model <model_identifier> --measure <measure> --context <context> --text <text>

    The entrypoint must return a JSON string to stdout with the following format based on the following tasks:

    Task: predict next word
    Input: --measure "next-word"
    Output: {"measure": NEXT_WORD} 
    where NEXT_WORD is a string

    Task: estimate reading times
    Input: --measure "token-logits"
    Output: {"tokens": TOKENS, "measure": LOGITS} 
    where LOGITS are the logits predicting the target tokens and TOKENS are the corresponding true token indices
    LOGITS are thus generated using the text up until the last token whereas TOKENS are the true tokens
    Both LOGITS and TOKENS should have the same shape[0], and shape[1] should be the vocabulary size for the LOGITS
    Objects should be returned as lists, e.g., object.cpu().numpy().tolist() for pytorch tensors

    Task: extract representation
    Input: --measure MEASURE
    Output: {"measure": REPRESENTATION}
    where MEASURE is the name of the representation as supported by your container 
    and REPRESENTATION is an array of shape (1, representation_size) cast to a list

    Note: While the internals of any containerized model are not restricted, the interface must be as described above. 
    It is highly recommended to raise detailed error messages from inside the container, so they can be escalated here.
    It is also recommended to include a list of supported measures in the container's documentation.

    Note: To add new tasks to this interface, open a PR to update the task mapping in the constructor of this class.
    The new entry to the task dict should map the task to a function that takes the output of the container and returns a score.
    See current task implementations for examples.
    """

    def __init__(
            self,
            container: str,
            entrypoint: str,
            identifier: str,
            region_layer_mapping: dict,
            task_heads: Union[None, Dict[ArtificialSubject.Task, Callable]] = None,
    ):
        """
        :param container: Container name, e.g., "USERNAME/CONTAINER:TAG"
        :param entrypoint: Entrypoint to run inside container, e.g., "python /path/to/entrypoint.py"
        :param identifier: Model identifer passed to entrypoint, e.g., "model_name"
        :param region_layer_mapping: Mapping from brain region to requested measure, e.g., {"language_system": "model_layer_name"}
        :param task_heads: Mapping from task to callable that takes the output of the container and returns a score, e.g., {ArtificialSubject.Task.next_word: predict_next_word_function}
        """
        self._logger = logging.getLogger(fullname(self))
        self._container: str = container
        self._entrypoint: str = entrypoint
        self._identifier: str = identifier
        self._region_layer_mapping: dict = region_layer_mapping

        self._neural_recordings: List[Tuple] = []
        self._behavioral_task: Union[None, ArtificialSubject.Task] = None
        task_mapping_default = {
            ArtificialSubject.Task.next_word: self._predict_next_word,
            ArtificialSubject.Task.reading_times: self._estimate_reading_times,
        }
        self._task_function_mapping_dict = (
            {**task_mapping_default, **task_heads}
            if task_heads
            else task_mapping_default
        )
        self._token_count = 0

        self._backend = self._select_container_backend()
        self._cachedir = Path.home() / ".cache" / "brainscore_language"
        self._cachedir.mkdir(parents=True, exist_ok=True)
        self._download_container()

    def identifier(self):
        return self._identifier

    def start_behavioral_task(self, task: ArtificialSubject.Task):
        self._behavioral_task = task
        self._behavioral_function = self._task_function_mapping_dict[task]

    def start_neural_recording(
        self,
        recording_target: ArtificialSubject.RecordingTarget,
        recording_type: ArtificialSubject.RecordingType,
    ):
        self._neural_recordings.append((recording_target, recording_type))

    def _select_container_backend(self):
        options = ["docker", "singularity"]
        for option in options:
            try:
                subprocess.run(
                    [option, "--version"], stdout=subprocess.DEVNULL
                )  # attempt to run the container backend, try another on error
                return option
            except:
                self._logger.info(f"{option} backend not found. Testing next option.")
        raise RuntimeError(
            f"Could not find any of the following container backends: {options}. Please install one."
        )

    @staticmethod
    def _get_singularity_container(cachedir: Path, container: str) -> Path:
        f = cachedir / f"{container.split('/')[1].replace(':', '_')}.sif"
        return f

    def _download_container(self):
        """
        Download container to cache directory if it does not exist yet.
        """

        # build command
        if self._backend == "docker":
            cmd = ["docker", "pull", f"{self._container}"]
        elif self._backend == "singularity":
            f = self._get_singularity_container(self._cachedir, self._container)
            if f.exists():
                self._logger.info(f"Container already downloaded to {self._cachedir}.")
                return
            cmd = ["singularity", "pull", f"docker://{self._container}"]
        else:
            raise RuntimeError(f"Unknown container backend {self._backend}")

        # run command
        try:
            process = subprocess.Popen(cmd, cwd=self._cachedir, stdout=subprocess.PIPE)
            for line in iter(process.stdout.readline, b""):
                self._logger.debug(line)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Could not pull container {self._container} using {self._backend}. Error message above traceback."
            ) from e

    def _evaluate_container(self, context: str, text: str, measure: str) -> np.ndarray:
        """
        Pass arguments to container and return results if interface is followed.
        If the container fails, the error message is escalated.
        """

        def prep(s):
            return re.sub(r"\s+", " ", s).replace('"', "'").replace("'", r"'\''")

        if self._backend == "docker":
            container = self._container
        elif self._backend == "singularity":
            container = self._get_singularity_container(self._cachedir, self._container)
        else:
            raise RuntimeError(f"Unknown container backend {self._backend}")

        cmd = f"""{self._backend} run {container} "{self._entrypoint} """
        cmd += f"""--model {self._identifier} --measure {measure} """
        cmd += f"""--context '{prep(context)}' --text '{prep(text)}' " """

        try:
            output = subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Error while running container: {e.output}")
            raise RuntimeError(
                f"Container {self._container} raised an error. "
                + "Please confirm it supports the requested interface and arguments."
            ) from e

        response_json = output.decode("utf-8")

        return json.loads(response_json)

    def _predict_next_word(self, context: str, text: str) -> str:
        output = self._evaluate_container(context, text, "next-word")
        next_word = output["measure"]
        assert isinstance(next_word, str)
        return next_word

    def _estimate_reading_times(self, context: str, text: str) -> float:
        import torch.nn.functional as F

        output = self._evaluate_container(context, text, "token-logits")
        shifted_logits = torch.Tensor(output["measure"])
        tokens = torch.Tensor(output["tokens"]).long()
        assert shifted_logits.shape[0] == tokens.shape[0]
        return F.cross_entropy(shifted_logits, tokens, reduction="sum") / np.log(2)

    def _record_representation(
            self, context: str, text: str, representation: str
    ) -> np.ndarray:
        output = self._evaluate_container(context, text, representation)
        representation = np.array(output["measure"])
        assert representation.shape[0] == 1
        return representation

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, DataAssembly]:
        """
        Same high-level structure as HuggingFace models,
        but parallelized over text parts for efficiency,
        due to longer delays associated with container evaluation.
        """

        def _build_assembly(part_number, text_part):
            context = " ".join(text[: part_number + 1])
            stimuli_coords = {
                "stimulus": ("presentation", [text_part]),
                "context": ("presentation", [context]),
                "part_number": ("presentation", [part_number]),
            }
            if self._behavioral_task:
                behavioral_output = self._behavioral_function(context, text_part)
                behavior = BehavioralAssembly(
                    [behavioral_output], coords=stimuli_coords, dims=["presentation"]
                )
                return "behavior", behavior
            if self._neural_recordings:
                representations = OrderedDict()
                for recording_target, recording_type in self._neural_recordings:
                    measure = self._region_layer_mapping[recording_target]
                    recording = self._record_representation(context, text_part, measure)
                    representations[
                        (recording_target, recording_type, measure)
                    ] = recording
                neural = self._build_neural_assembly(representations, stimuli_coords)
                return ("neural", neural)

        if type(text) == str:
            text = [text]
        text_iterator = tqdm(text, desc="digest text") if len(text) > 100 else text
        with parallel_backend("loky", n_jobs=multiprocessing.cpu_count()):
            assemblies = Parallel()(
                delayed(_build_assembly)(part_number, text_part)
                for part_number, text_part in enumerate(text_iterator)
            )

        self._logger.debug("Merging outputs")
        output = {"behavior": [], "neural": []}
        for output_type, assembly in assemblies:
            output[output_type].append(assembly)
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

    @staticmethod
    def _build_neural_assembly(representations, stimuli_coords):
        representation_values = np.concatenate(
            [values for values in representations.values()], axis=-1
        )
        neuroid_coords = {
            "layer": (
                "neuroid",
                np.concatenate(
                    [
                        [measure] * values.shape[-1]
                        for (_, _, measure), values in representations.items()
                    ]
                ),
            ),
            "region": (
                "neuroid",
                np.concatenate(
                    [
                        [target] * values.shape[-1]
                        for (target, _, _), values in representations.items()
                    ]
                ),
            ),
            "recording_type": (
                "neuroid",
                np.concatenate(
                    [
                        [rtype] * values.shape[-1]
                        for (_, rtype, _), values in representations.items()
                    ]
                ),
            ),
            "neuron_number_in_layer": (
                "neuroid",
                np.concatenate(
                    [np.arange(values.shape[-1]) for values in representations.values()]
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
        neural = NeuroidAssembly(
            representation_values,
            coords={**stimuli_coords, **neuroid_coords},
            dims=["presentation", "neuroid"],
        )
        return neural
