from abc import ABC
import typing

import xarray as xr

from langbrainscore.interface.cacheable import _Cacheable
from langbrainscore.utils.xarray import fix_xr_dtypes


class _Dataset(_Cacheable, ABC):
    """
    wrapper class for xarray DataArray that confirms format adheres to interface.
    """

    dataset_name: str = None

    def __init__(
        self,
        xr_obj: xr.DataArray,
        dataset_name: str = None,
        # modality: str = None,
        _skip_checks: bool = False,
    ) -> "_Dataset":
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
                self.dataset_name = xr_obj.attrs["name"]
            except KeyError:
                pass
        self.dataset_name = self.dataset_name or dataset_name
        # self.modality = modality

        if not _skip_checks:
            dims = ("sampleid", "neuroid", "timeid")
            coords = dims + ("stimulus", "subject")
            assert isinstance(xr_obj, xr.DataArray)
            assert xr_obj.ndim == len(dims)
            assert all([dim in xr_obj.dims for dim in dims])
            assert all([coord in xr_obj.coords for coord in coords])

        self._xr_obj = fix_xr_dtypes(xr_obj)

    # def __getattr__(self, __name: str) -> typing.Any:
    #     """falls back on the xarray object in case of a NameError using __getattribute__
    #     on this object"""
    #     try:
    #         return getattr(self.contents, __name)
    #     except AttributeError:
    #         raise AttributeError(f"no attribute called `{__name}` on object")
