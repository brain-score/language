import xarray as xr
from langbrainscore.interface.cacheable import _Cacheable
from abc import ABC

class _Dataset(_Cacheable, ABC):
    """
    wrapper class for xarray DataArray that confirms format adheres to interface.
    """
    dataset_name: str = None

    def __init__(self, xr_obj: xr.DataArray, dataset_name: str = None, _skip_checks: bool = False) -> "_Dataset":
        """
        accepts an xarray with the following core
        dimensions: sampleid, neuroid, timeid
        and at least the following core
        coordinates: sampleid, neuroid, timeid, stimulus, subject

        Args:
            xr_obj (xr.DataArray): xarray object with core dimensions and coordinates
        """

        if xr_obj is not None:
            try:
                self.dataset_name = xr_obj.attrs['name']
            except KeyError:
                pass
        self.dataset_name = self.dataset_name or dataset_name

        if not _skip_checks:
            dims = ("sampleid", "neuroid", "timeid")
            coords = dims + ("stimulus", "subject")
            assert isinstance(xr_obj, xr.DataArray)
            assert xr_obj.ndim == len(dims)
            assert all([dim in xr_obj.dims for dim in dims])
            assert all([coord in xr_obj.coords for coord in coords])
            self._xr_obj = xr_obj
