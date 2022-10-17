import functools
import json
import logging
import multiprocessing
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict, Union, Callable

import numpy as np
import torch
import xarray as xr
from joblib import Parallel, delayed, parallel_backend
from numpy.core import defchararray
from tqdm import tqdm

from brainio.assemblies import DataAssembly, NeuroidAssembly, BehavioralAssembly
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils import fullname


class ContainerSubject(ArtificialSubject):
    def __init__(
        self,
        container: str,
        entrypoint: str,
        identifier: str,
        region_layer_mapping: dict,
        task_heads: Union[None, Dict[ArtificialSubject.Task, Callable]] = None,
    ):
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

    def perform_behavioral_task(self, task: ArtificialSubject.Task):
        self._behavioral_task = task
        self._behavioral_function = self._task_function_mapping_dict[task]

    def perform_neural_recording(
        self,
        recording_target: ArtificialSubject.RecordingTarget,
        recording_type: ArtificialSubject.RecordingType,
    ):
        self._neural_recordings.append((recording_target, recording_type))

    def _select_container_backend(self):
        options = ["docker", "singularity"]
        for option in options:
            try:
                subprocess.run([option, "--version"], stdout=subprocess.DEVNULL)
                return option
            except:
                self._logger.info(f"{option} backend not found. Testing next option.")
        raise RuntimeError(
            f"Could not find any of the following container backends: {options}. Please install one."
        )

    @staticmethod
    def _get_singularity_container(cachedir: Path, container: str) -> str:
        f = cachedir / f"{container.split('/')[1].replace(':','_')}.sif"
        return f

    def _download_container(self):
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
        process = subprocess.Popen(cmd, cwd=self._cachedir, stdout=subprocess.PIPE)
        for line in iter(process.stdout.readline, b""):
            self._logger.debug(line)
        if not f.exists():
            raise RuntimeError(
                f"Could not pull container {self._container} using {self._backend}. Error message above traceback."
            )

    def _evaluate_container(self, context: str, text: str, measure: str) -> np.ndarray:
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
        return output["measure"]

    def _estimate_reading_times(self, context: str, text: str) -> float:
        import torch.nn.functional as F

        output = self._evaluate_container(context, text, "token-logits")
        shifted_logits = torch.Tensor(output["measure"])
        tokens = torch.Tensor(output["tokens"]).long()
        return F.cross_entropy(shifted_logits, tokens, reduction="sum") / np.log(2)

    def _record_representation(
        self, context: str, text: str, representation: str
    ) -> np.ndarray:
        output = self._evaluate_container(context, text, representation)
        return np.array(output["measure"])

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, DataAssembly]:
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
                return ("behavior", behavior)
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
        text_iterator = tqdm(text, desc="digest text") if len(text) > 1 else text
        with parallel_backend("loky", n_jobs=multiprocessing.cpu_count()):
            assemblies = Parallel()(
                delayed(_build_assembly)(part_number, text_part)
                for part_number, text_part in enumerate(text_iterator)
            )

        self._logger.debug("Merging outputs")
        output = {"behavior": [], "neural": []}
        for assembly in assemblies:
            output[assembly[0]].append(assembly[1])
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
