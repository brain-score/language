import numpy as np
from scipy.stats import pearsonr

from brainio.assemblies import DataAssembly
from brainscore_core.metrics import Score, Metric
from brainscore_language import metric_registry


class PearsonCorrelation(Metric):
    """
    Pearson-r correlation between two vectors.
    """

    def __call__(self, assembly1: DataAssembly, assembly2: DataAssembly) -> Score:
        rvalue, pvalue = pearsonr(assembly1, assembly2)
        score = Score(np.abs(rvalue))
        score.attrs['rvalue'] = rvalue
        score.attrs['pvalue'] = pvalue
        return score


metric_registry['pearsonr'] = PearsonCorrelation
