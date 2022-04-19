import xarray as xr
from sklearn.impute import SimpleImputer


def copy_metadata(target: xr.DataArray, source: xr.DataArray, dim: str) -> xr.DataArray:
    """copies the metadata coordinates of a source xarray on dimension `dim` over to target xarray
        for this to work, the two `xr.DataArray` objects must have the same dimensions and
        dimensionality of data, minimally in the `dim` dimension.
        
    Args:
        target (xr.DataArray): target xarray to copy the metadata coordinates onto
            (a copy is made, this does not happen inplace)
        source (xr.DataArray): the source xarray for the metadata coordinates along `dim`
        dim (str): dimension of the metadata coordinates (see `xr` documentation for help)

    Returns:
        xr.DataArray
    """
    for coord in source[dim].coords:
        target = target.assign_coords({coord: (dim, source[coord].data)})
    return target


def collapse_multidim_coord(xr_obj, coord, keep_dim):
    """As a result of iterative construction of `xarray`s in our various functions
        (such as in the `HuggingFaceEncoder.encode` method), the same values are repeated
        over and over again

    Args:
        xr_obj (xr.Array): _description_
        coord (str): _description_
        keep_dim (bool): _description_

    Returns:
        _type_: _description_
    """
    imputer = SimpleImputer(strategy="most_frequent")
    try:
        stimuli = imputer.fit_transform(xr_obj[coord])[0]
        return xr_obj.assign_coords({coord: (keep_dim, stimuli)})
    except ValueError as e: # TODO which exception? what scenario does this cover?
        stimuli = imputer.fit_transform(xr_obj[coord]).transpose()[0]
        return xr_obj.assign_coords({coord: (keep_dim, stimuli)})
