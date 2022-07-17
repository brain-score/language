import hashlib
import logging
from pathlib import Path

import entrypoints
import numpy as np
import pandas as pd
from brainio.catalogs import Catalog, SOURCE_CATALOG

ENTRYPOINT = "brainio_lookups"
TYPE_ASSEMBLY = 'assembly'
TYPE_STIMULUS_SET = 'stimulus_set'
_catalogs = {}

_logger = logging.getLogger(__name__)


def list_catalogs():
    return sorted(list(entrypoints.get_group_named(ENTRYPOINT).keys()))


def _load_catalog(identifier, entry_point):
    catalog = entry_point.load()()
    assert isinstance(catalog, Catalog)
    assert catalog.identifier == identifier
    return catalog


def _load_installed_catalogs():
    installed_catalogs = entrypoints.get_group_named(ENTRYPOINT)
    _logger.debug(f"Loading catalog from entrypoints")
    print(f"Loading catalog from entrypoints")
    for k, v in installed_catalogs.items():
        catalog = _load_catalog(k, v)
        _catalogs[k] = catalog
    return _catalogs


def get_catalog(identifier):
    catalogs = get_catalogs()
    return catalogs[identifier]


def get_catalogs():
    if not _catalogs:
        _load_installed_catalogs()
    return _catalogs


def combined_catalog():
    catalogs = get_catalogs()
    for identifier, catalog in catalogs.items():
        catalog[SOURCE_CATALOG] = identifier
    concat_catalogs = pd.concat(catalogs.values(), ignore_index=True)
    return concat_catalogs


def list_stimulus_sets():
    combined = combined_catalog()
    stimuli_rows = combined[combined['lookup_type'] == TYPE_STIMULUS_SET]
    return sorted(list(set(stimuli_rows['identifier'])))


def list_assemblies():
    combined = combined_catalog()
    assembly_rows = combined[combined['lookup_type'] == TYPE_ASSEMBLY]
    return sorted(list(set(assembly_rows['identifier'])))


def lookup_stimulus_set(identifier):
    combined = combined_catalog()
    lookup = combined[(combined['identifier'] == identifier) & (combined['lookup_type'] == TYPE_STIMULUS_SET)]
    if len(lookup) == 0:
        raise StimulusSetLookupError(f"Stimulus set {identifier} not found")
    csv_lookup = _lookup_stimulus_set_filtered(lookup, filter_func=_is_csv_lookup, label="CSV")
    zip_lookup = _lookup_stimulus_set_filtered(lookup, filter_func=_is_zip_lookup, label="ZIP")
    return csv_lookup, zip_lookup


def _lookup_stimulus_set_filtered(lookup, filter_func, label):
    cols = [n for n in lookup.columns if n != SOURCE_CATALOG]
    # filter for csv vs. zip
    # if there are any groups of rows where every field except source is the same,
    # we only want one from each group
    filtered_rows = lookup[lookup.apply(filter_func, axis=1)].drop_duplicates(subset=cols)
    identifier = lookup.iloc[0]['identifier']
    if len(filtered_rows) == 0:
        raise StimulusSetLookupError(f"{label} for stimulus set {identifier} not found")
    if len(filtered_rows) > 1: # there were multiple rows but not all identical
        raise RuntimeError(
            f"Internal data inconsistency: Found more than 2 lookup rows for stimulus_set {label} for identifier {identifier}")
    assert len(filtered_rows) == 1
    return filtered_rows.squeeze()


def lookup_assembly(identifier):
    combined = combined_catalog()
    lookup = combined[(combined['identifier'] == identifier) & (combined['lookup_type'] == TYPE_ASSEMBLY)]
    if len(lookup) == 0:
        raise AssemblyLookupError(f"Data assembly {identifier} not found")
    cols = [n for n in lookup.columns if n != SOURCE_CATALOG]
    # if there are any groups of rows where every field except source is the same,
    # we only want one from each group
    de_dupe = lookup.drop_duplicates(subset=cols)
    if len(de_dupe) > 1: # there were multiple rows but not all identical
        raise RuntimeError(f"Internal data inconsistency: Found multiple lookup rows for identifier {identifier}")
    assert len(de_dupe) == 1
    return de_dupe.squeeze()


class StimulusSetLookupError(KeyError):
    pass


class AssemblyLookupError(KeyError):
    pass


def append(catalog_identifier, object_identifier, cls, lookup_type,
           bucket_name, sha1, s3_key, stimulus_set_identifier=None):
    catalogs = get_catalogs()
    catalog = catalogs[catalog_identifier]
    catalog_path = catalog.source_path
    _logger.debug(f"Adding {lookup_type} {object_identifier} to catalog {catalog_identifier}")
    object_lookup = {
        'identifier': object_identifier,
        'lookup_type': lookup_type,
        'class': cls,
        'location_type': "S3",
        'location': f"https://{bucket_name}.s3.amazonaws.com/{s3_key}",
        'sha1': sha1,
        'stimulus_set_identifier': stimulus_set_identifier,
        'lookup_source': catalog_identifier,
    }
    # check duplicates
    assert object_lookup['lookup_type'] in [TYPE_ASSEMBLY, TYPE_STIMULUS_SET]
    duplicates = catalog[(catalog['identifier'] == object_lookup['identifier']) &
                           (catalog['lookup_type'] == object_lookup['lookup_type'])]
    if len(duplicates) > 0:
        if object_lookup['lookup_type'] == TYPE_ASSEMBLY:
            raise ValueError(f"Trying to add duplicate identifier {object_lookup['identifier']}, "
                             f"existing \n{duplicates.to_string()}")
        elif object_lookup['lookup_type'] == TYPE_STIMULUS_SET:
            if len(duplicates) == 1 and duplicates.squeeze()['identifier'] == object_lookup['identifier'] and (
                    (_is_csv_lookup(duplicates.squeeze()) and _is_zip_lookup(object_lookup)) or
                    (_is_zip_lookup(duplicates.squeeze()) and _is_csv_lookup(object_lookup))):
                pass  # all good, we're just adding the second part of a stimulus set
            else:
                raise ValueError(
                    f"Trying to add duplicate identifier {object_lookup['identifier']}, existing {duplicates}")
    # append and save
    add_lookup = pd.DataFrame({key: [value] for key, value in object_lookup.items()})
    catalog = catalog.append(add_lookup)
    catalog.to_csv(catalog_path, index=False)
    _catalogs[catalog_identifier] = catalog
    return catalog


def _is_csv_lookup(data_row):
    return data_row['lookup_type'] == TYPE_STIMULUS_SET \
           and data_row['location'].endswith('.csv') \
           and data_row['class'] not in [None, np.nan]


def _is_zip_lookup(data_row):
    return data_row['lookup_type'] == TYPE_STIMULUS_SET \
           and data_row['location'].endswith('.zip') \
           and data_row['class'] in [None, np.nan]


def sha1_hash(path, buffer_size=64 * 2 ** 10):
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        buffer = f.read(buffer_size)
        while len(buffer) > 0:
            sha1.update(buffer)
            buffer = f.read(buffer_size)
    return sha1.hexdigest()
