import scipy.stats
import numpy as np
import typing
from tqdm import tqdm
from langbrainscore.interface.metrics import _Metric
import xarray as xr

class Metric(_Metric):

    def __init__(self, metric: typing.Union[str, typing.Callable]) -> None:
        self.metric = metric

    def __call__(self, A: xr.DataArray, B: xr.DataArray, **kwds: dict) -> np.ndarray:
        if A.shape != B.shape:
            raise ValueError(f'mismatched shapes of A, B:  {A.shape}, {B.shape}')
        if len(A.shape) < 2:
            raise ValueError
        # return np.apply_along_axis(self.metric, 1, )
        return [self.metric(A[:, i], B[:, i], **kwds) 
                for i in tqdm(range(A.shape[1]), desc='computing metric per neuroid in a cvfold')]



def pearson_r(x, y):
    """
    Calculates the Pearson correlation coefficient between two lists of
    numbers.

    Parameters
    ----------
    x : list
        List of numbers.
    y : list
        List of numbers.

    Returns
    -------
    float
        Pearson correlation coefficient.

    Raises
    ------
    ValueError
        If lists are not of equal length.
    """
    if len(x) != len(y):
        raise ValueError("Lists must be of equal length.")

    r, p = scipy.stats.pearsonr(x, y)
    return r

def pearson_r_nd(X, Y):
    assert X.shape == Y.shape
    return np.diag(np.corrcoef(X.T,Y.T)[X.shape[1]:,:X.shape[1]])
