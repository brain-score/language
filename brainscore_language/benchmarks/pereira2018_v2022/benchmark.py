from brainio.assemblies import NeuroidAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_language.utils.s3 import load_from_s3
from brainscore_language import load_dataset, load_metric

import logging

logger = logging.getLogger(__name__)

BIBTEX = """@article{pereira2018toward,
  title={Toward a universal decoder of linguistic meaning from brain activation},
  author={Pereira, Francisco and Lou, Bin and Pritchett, Brianna and Ritter, Samuel and Gershman, Samuel J 
          and Kanwisher, Nancy and Botvinick, Matthew and Fedorenko, Evelina},
  journal={Nature communications},
  volume={9},
  number={1},
  pages={1--13},
  year={2018},
  publisher={Nature Publishing Group}
}"""


def Pereira2018_243sentences():
    return _Pereira2018LinregPearsonr(
        experiment="PereiraE3_72pass",
        ceiling_s3_kwargs=dict(
            sha1="64ebaded266506df698ed249685ffadabf017951",
            version_id="H8Apri0513nJ8Qy54dp3IitpuLOL9UyK",
            raw_kwargs=dict(
                version_id="ws6mzNrk9AkN2rFP3vVzBcSSgF3n0J5O",
                sha1="a72adbc813e1eaff5fbb3e4fcb9c1b202d3cdbf2",
                raw_kwargs=dict(
                    version_id="LYWHrqobBWLYYKMwMz4bEUsB2s..9QTY",
                    sha1="0b5477ead370939a65b824629f72fad40da0543f",
                ),
            ),
        ),
    )


def Pereira2018_384sentences():
    return _Pereira2018LinregPearsonr(
        experiment="PereiraE2_96pass",
        ceiling_s3_kwargs=dict(
            sha1="638b7a0d7a08a851ace19af9489801c9c1a4c05b",
            version_id="OFu.5imbaIbrJvD6xny_G55XfEzOU37o",
            raw_kwargs=dict(
                sha1="3348a8c489f5d091c5f4ded9d50f0c895e2368f4",
                version_id="c6zAAvSRxerQ3kHRf0I7WnSoFZDSd3kG",
                raw_kwargs=dict(
                    sha1="75a88348f30135bb0453459ba8eddf33eabf49ff",
                    version_id="cA3D7oKIcpx.qZz9tLOFO_kAkOduht4W",
                ),
            ),
        ),
    )


class _Pereira2018LinregPearsonr(BenchmarkBase):
    """
    Evaluate model ability to predict neural activity in the human language system in response to natural sentences,
    recorded by Pereira et al. 2018.
    Alignment of neural activity between model and human subjects is evaluated via cross-validated linear predictivity.

    This benchmark builds off the behavioral benchmark introduced
    in Schrimpf et al. 2021 (https://www.pnas.org/doi/10.1073/pnas.2105646118), but:

    * computes neural alignment to each of the two experiments ({243,384}sentences) separately, as well as ceilings
    * requires the model to have committed to neural readouts (e.g. layer 41 corresponds to the language system),
        rather than testing every layer separately

    Each of these benchmarks evaluates one of the two experiments, the overall Pereira2018-linear score is the mean of
    the two ceiling-normalized scores.
    """

    def __init__(
        self,
        experiment: str,
        ceiling_s3_kwargs: dict,
    ):
        self.data = self._load_data(experiment)
        self.metric = load_metric(
            "linreg_pearsonr",
            crossvalidation_kwargs={
                "split_coord": "sentence",
            },
            regression_kwargs={
                "stimulus_coord": "sentence",
            },
            correlation_kwargs={
                "correlation_coord": "sentence",
                "neuroid_coord": "neuroid",
            },
        )
        identifier = f"Pereira2018_v2022.{experiment}-linreg_pearsonr"
        ceiling = self._load_ceiling(identifier=identifier, **ceiling_s3_kwargs)
        super(_Pereira2018LinregPearsonr, self).__init__(
            identifier=identifier,
            ceiling_func=ceiling,
            version=1,
            parent="Pereira2018-linear",
            bibtex=BIBTEX,
        )

    def _load_data(self, experiment: str) -> NeuroidAssembly:
        data = load_dataset("Pereira2018_v2022.language")
        # data = data.sel(experiment=experiment)  # filter experiment
        data.loc[data.experiment == experiment, :, :]
        data = data.dropna(
            "neuroid"
        )  # not all subjects have done both experiments, drop those that haven't
        data.attrs["identifier"] = f"{data.identifier}.{experiment}"
        if "time" in data.dims:
            data = data.drop("time").squeeze("time")
        return data

    def _load_ceiling(
        self,
        identifier: str,
        version_id: str,
        sha1: str,
        assembly_prefix="ceiling_",
        raw_kwargs=None,
    ):
        ceiling = load_from_s3(
            identifier,
            cls=Score,
            assembly_prefix=assembly_prefix,
            version_id=version_id,
            sha1=sha1,
        )
        if raw_kwargs:  # recursively load raw attributes
            raw = self._load_ceiling(
                identifier=identifier,
                assembly_prefix=assembly_prefix + "raw_",
                **raw_kwargs,
            )
            ceiling.attrs["raw"] = raw
        return ceiling

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.perform_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system,
            recording_type=ArtificialSubject.RecordingType.fMRI,
        )
        stimuli = self.data["stimuli"]
        predictions = candidate.digest_text(stimuli.values)["neural"]
        predictions["presentation"] = "presentation", stimuli["sentence"].values
        raw_score = self.metric(
            predictions,
            self.data,
        )
        score = raw_score  # ceiling_normalize(raw_score, self.ceiling)
        return score
