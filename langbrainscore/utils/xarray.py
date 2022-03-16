import xarray as xr


def copy_metadata(target, source, dim):
    for coord in source[dim].coords:
        target = target.assign_coords({coord: (dim, source[coord].data)})
    return target
