import numpy as np

from brainscore_core.supported_data_standards.brainio.assemblies import walk_coords


def fullname(obj):
    """ Resolve the full module-qualified name of an object. Typically used for logger naming. """
    return obj.__module__ + "." + obj.__class__.__name__


def attach_presentation_meta(assembly, meta):
    for coord, dims, values in walk_coords(meta):
        if len(dims) != 1 or np.array(dims).item() != 'presentation':  # dimension mismatch
            continue
        if hasattr(assembly, coord):  # coordinate already part of assembly
            continue
        assembly[coord] = (dims, values)
