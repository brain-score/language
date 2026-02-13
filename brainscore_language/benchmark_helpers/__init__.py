import numpy as np
from typing import List

from brainscore_core.supported_data_standards.brainio.assemblies import walk_coords, array_is_element, DataAssembly


def ci_error(samples, center, confidence=.95):
    low, high = 100 * ((1 - confidence) / 2), 100 * (1 - ((1 - confidence) / 2))
    confidence_below, confidence_above = np.nanpercentile(samples, low), np.nanpercentile(samples, high)
    confidence_below, confidence_above = center - confidence_below, confidence_above - center
    return confidence_below, confidence_above


def manual_merge(*elements: List[DataAssembly], on='neuroid') -> DataAssembly:
    """
    Manually merge a set of assemblies where xarray's automated merge might fail.
    This function likely covers only covers a small number of use-cases, and should thus be used with caution.
    """
    dims = elements[0].dims
    assert all(element.dims == dims for element in elements[1:])
    merge_index = dims.index(on)
    # the coordinates in the merge index should have the same keys
    assert _coords_match(elements, dim=on,
                         match_values=False), f"coords in {[element[on] for element in elements]} do not match"
    # all other dimensions, their coordinates and values should already align
    for dim in set(dims) - {on}:
        assert _coords_match(elements, dim=dim,
                             match_values=True), f"coords in {[element[dim] for element in elements]} do not match"
    # merge values without meta
    merged_values = np.concatenate([element.values for element in elements], axis=merge_index)
    # piece together with meta
    result = type(elements[0])(merged_values, coords={
        **{coord: (dims, values)
           for coord, dims, values in walk_coords(elements[0])
           if not array_is_element(dims, on)},
        **{coord: (dims, np.concatenate([element[coord].values for element in elements]))
           for coord, dims, _ in walk_coords(elements[0])
           if array_is_element(dims, on)}}, dims=elements[0].dims)
    return result


def _coords_match(elements, dim, match_values=False):
    """ Helper method for `manual_merge` """
    first_coords = [(key, tuple(value)) if match_values else key for _, key, value in walk_coords(elements[0][dim])]
    other_coords = [[(key, tuple(value)) if match_values else key for _, key, value in walk_coords(element[dim])]
                    for element in elements[1:]]
    return all(tuple(first_coords) == tuple(coords) for coords in other_coords)
