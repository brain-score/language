import logging

from brainio.assemblies import NeuroidAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric, benchmark_registry
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_language.utils.s3 import load_from_s3

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
    return _Pereira2018LinregPearsonr(experiment="PereiraE2_96pass")


def Pereira2018_384sentences():
    return _Pereira2018LinregPearsonr(experiment="PereiraE3_72pass")


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
        # ceiling_s3_kwargs: dict = None,
    ):
        self.data = self._load_data(experiment)
        self.metric = load_metric("linear_pearsonr")
        identifier = f"Pereira2018.{experiment}-linear"
        # ceiling = self._load_ceiling(identifier=identifier, **ceiling_s3_kwargs)
        super(_Pereira2018LinregPearsonr, self).__init__(
            identifier=identifier,
            version=1,
            parent="Pereira2018-linear",
            ceiling=None,  # ceiling,
            bibtex=BIBTEX,
        )

    def _load_data(self, experiment: str) -> NeuroidAssembly:
        data = load_dataset("Pereira2018_v2022.language")
        data = data.sel(experiment=experiment)  # filter experiment
        data = data.dropna(
            "neuroid"
        )  # not all subjects have done both experiments, drop those that haven't
        data.attrs["identifier"] = f"{data.identifier}.{experiment}"
        return data

    # def _load_ceiling(
    #     self,
    #     identifier: str,
    #     version_id: str,
    #     sha1: str,
    #     assembly_prefix="ceiling_",
    #     raw_kwargs=None,
    # ):
    #     ceiling = load_from_s3(
    #         identifier,
    #         cls=Score,
    #         assembly_prefix=assembly_prefix,
    #         version_id=version_id,
    #         sha1=sha1,
    #     )
    #     if raw_kwargs:  # recursively load raw attributes
    #         raw = self._load_ceiling(
    #             identifier=identifier,
    #             assembly_prefix=assembly_prefix + "raw_",
    #             **raw_kwargs,
    #         )
    #         ceiling.attrs["raw"] = raw
    #     return ceiling

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.perform_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system,
            recording_type=ArtificialSubject.RecordingType.fMRI,
        )
        stimuli = self.data["sentence"]
        predictions = candidate.digest_text(stimuli.values)["neural"]
        predictions["stimulus_id"] = "presentation", stimuli["stimulus_id"].values
        raw_score = self.metric(predictions, self.data)
        score = raw_score  # ceiling_normalize(raw_score, self.ceiling)
        return score


benchmark_registry["Pereira2018_v2022.243sentences-linear"] = Pereira2018_243sentences
benchmark_registry["Pereira2018_v2022.384sentences-linear"] = Pereira2018_384sentences
