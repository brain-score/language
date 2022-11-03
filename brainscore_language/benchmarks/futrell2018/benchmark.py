import logging
import numpy as np
from numpy.random import RandomState

from brainio.assemblies import DataAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score, Metric
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils import attach_presentation_meta
from brainscore_language.utils.ceiling import ceiling_normalize

logger = logging.getLogger(__name__)


class Futrell2018Pearsonr(BenchmarkBase):
    """
    Evaluate model ability to predict reading times on the natural stories corpus introduced in Futrell et al. 2018.
    Alignment of reading times between model and human subjects is evaluated via Pearson correlation.

    This benchmark builds off the behavioral benchmark introduced in Schrimpf et al. 2021, but:

    * does not allow for any fitting; rather model candidates have to directly output reading times, and
    * estimates the ceiling with Spearman-Brown corrected split-halves consistency.
    """

    def __init__(self):
        self.data = load_dataset("Futrell2018")
        self.metric = load_metric("pearsonr")
        ceiler = SplitHalvesConsistency(
            num_splits=10, split_coordinate="subject_id", consistency_metric=self.metric
        )
        ceiling = ceiler(self.data)
        super(Futrell2018Pearsonr, self).__init__(
            identifier="Futrell2018-pearsonr",
            version=1,
            parent="behavior",
            ceiling_func=ceiling,
            bibtex=self.data.bibtex,
        )

    def __call__(self, candidate: ArtificialSubject) -> Score:
        # run experiment
        candidate.start_behavioral_task(ArtificialSubject.Task.reading_times)
        stimuli = self.data["word"].values
        predictions = candidate.digest_text(stimuli)["behavior"]
        attach_presentation_meta(predictions, self.data["presentation"])
        # exclude first words
        predictions = predictions[predictions["word_within_sentence_id"] != 1]
        targets = self.data[self.data["word_within_sentence_id"] != 1]
        targets = targets.mean("subject")  # compare to "average human"
        # score
        raw_score = self.metric(predictions, targets)
        score = ceiling_normalize(raw_score, self.ceiling)
        return score


class SplitHalvesConsistency:
    # following
    # https://github.com/brain-score/brain-score/blob/c51b8aa2c94212a9ac56c06c556afad0bb0a3521/brainscore/metrics/ceiling.py#L25-L96

    def __init__(
        self, num_splits: int, split_coordinate: str, consistency_metric: Metric
    ):
        """
        :param num_splits: how many times to create two halves
        :param split_coordinate: over which coordinate to split the assembly into halves
        :param consistency_metric: which metric to use to compute the consistency of two halves
        """
        self.num_splits = num_splits
        self.split_coordinate = split_coordinate
        self.consistency_metric = consistency_metric

    def __call__(self, assembly: DataAssembly) -> Score:
        split_dim = np.array(assembly[self.split_coordinate].dims).item()
        split_values = assembly[self.split_coordinate].values
        random_state = RandomState(0)
        consistencies, uncorrected_consistencies = [], []
        splits = range(self.num_splits)
        for _ in splits:
            half1_values = random_state.choice(
                split_values, size=len(split_values) // 2, replace=False
            )
            half2_values = set(split_values) - set(
                half1_values
            )  # this only works because of `replace=False` above
            half1 = assembly[
                {split_dim: [value in half1_values for value in split_values]}
            ].mean(split_dim)
            half2 = assembly[
                {split_dim: [value in half2_values for value in split_values]}
            ].mean(split_dim)
            consistency = self.consistency_metric(half1, half2)
            uncorrected_consistencies.append(consistency)
            # Spearman-Brown correction for sub-sampling
            corrected_consistency = 2 * consistency / (1 + (2 - 1) * consistency)
            consistencies.append(corrected_consistency)
        consistencies = Score(consistencies, coords={"split": splits}, dims=["split"])
        uncorrected_consistencies = Score(
            uncorrected_consistencies, coords={"split": splits}, dims=["split"]
        )
        average_consistency = consistencies.median("split")
        average_consistency.attrs["raw"] = consistencies
        average_consistency.attrs[
            "uncorrected_consistencies"
        ] = uncorrected_consistencies
        return average_consistency
