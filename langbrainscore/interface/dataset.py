import xarray as xr


class _Dataset:
    """
    wrapper class for xarray DataArray that confirms format adheres to interface.
    """

    def __init__(self, xr_obj: xr.DataArray) -> "_Dataset":
        """
        accepts an xarray with the following core
        dimensions: sampleid, neuroid, timeid
        and at least the following core
        coordinates: sampleid, neuroid, timeid, stimulus, subject

        Args:
            xr_obj (xr.DataArray): xarray object with core dimensions and coordinates
        """
        dims = ("sampleid", "neuroid", "timeid")
        coords = dims + ("stimulus", "subject")
        assert isinstance(xr_obj, xr.DataArray)
        assert xr_obj.ndim == len(dims)
        assert all([dim in xr_obj.dims for dim in dims])
        assert all([coord in xr_obj.coords for coord in coords])
        self._xr_obj = xr_obj
