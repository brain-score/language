import xarray as xr
from sklearn.impute import SimpleImputer


def copy_metadata(target, source, dim):
    for coord in source[dim].coords:
        target = target.assign_coords({coord: (dim, source[coord].data)})
    return target


def collapse_multidim_coord(xr_obj, coord, keep_dim):
    imputer = SimpleImputer(strategy="most_frequent")
    try:
        stimuli = imputer.fit_transform(xr_obj[coord])[0]
        return xr_obj.assign_coords({coord: (keep_dim, stimuli)})
    except ValueError as e: # TODO which exception? what scenario does this cover?
        stimuli = imputer.fit_transform(xr_obj[coord]).transpose()[0]
        return xr_obj.assign_coords({coord: (keep_dim, stimuli)})
