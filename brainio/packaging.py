import logging
import os
import zipfile
from pathlib import Path
import re
import mimetypes

import boto3
import pandas as pd
import xarray as xr
from tqdm import tqdm
import numpy as np
from PIL import Image

import brainio.assemblies
from brainio import lookup, list_stimulus_sets, fetch
from brainio.fetch import resolve_assembly_class
from brainio.lookup import TYPE_ASSEMBLY, TYPE_STIMULUS_SET, sha1_hash
from xarray import DataArray

_logger = logging.getLogger(__name__)


def create_stimulus_zip(proto_stimulus_set, target_zip_path):
    """
    Create zip file for stimuli in StimulusSet.
    Files in the zip will follow a flat directory structure with each row's filename equal to the `stimulus_id` by default,
        or `stimulus_path_within_store` if passed.
    :param proto_stimulus_set: a `StimulusSet` with a `get_stimulus: stimulus_id -> local path` method, a `stimulus_id` column,
        and optionally a `stimulus_path_within_store` column.
    :param target_zip_path: path to write the zip file to
    :return: SHA1 hash of the zip file
    """
    _logger.debug(f"Zipping stimulus set to {target_zip_path}")
    os.makedirs(os.path.dirname(target_zip_path), exist_ok=True)
    arcnames = []
    with zipfile.ZipFile(target_zip_path, 'w') as target_zip:
        for _, row in proto_stimulus_set.iterrows():  # using iterrows instead of itertuples for very large StimulusSets
            stimulus_path = proto_stimulus_set.get_stimulus(row['stimulus_id'])
            extension = os.path.splitext(stimulus_path)[1]
            arcname = row['stimulus_path_within_store'] if hasattr(row, 'stimulus_path_within_store') else row['stimulus_id']
            arcname = arcname + extension
            target_zip.write(stimulus_path, arcname=arcname)
            arcnames.append(arcname)
    sha1 = sha1_hash(target_zip_path)
    return sha1, arcnames


def upload_to_s3(source_file_path, bucket_name, target_s3_key):
    _logger.debug(f"Uploading {source_file_path} to {bucket_name}/{target_s3_key}")

    file_size = os.path.getsize(source_file_path)
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="upload to s3") as progress_bar:
        def progress_hook(bytes_amount):
            if bytes_amount > 0:  # at the end, this sometimes passes a negative byte amount which tqdm can't handle
                progress_bar.update(bytes_amount)

        client = boto3.client('s3')
        client.upload_file(str(source_file_path), bucket_name, target_s3_key, Callback=progress_hook)


def extract_specific(proto_stimulus_set):
    general = ['stimulus_current_local_file_path', 'stimulus_path_within_store']
    stimulus_set_specific_attributes = set(proto_stimulus_set.columns) - set(general)
    return list(stimulus_set_specific_attributes)


def create_stimulus_csv(proto_stimulus_set, target_path):
    _logger.debug(f"Writing csv to {target_path}")
    specific_columns = extract_specific(proto_stimulus_set)
    specific_stimulus_set = proto_stimulus_set[specific_columns]
    specific_stimulus_set.to_csv(target_path, index=False)
    sha1 = sha1_hash(target_path)
    return sha1


def check_naming_convention(name):
    assert re.match(r"[a-z]+\.[A-Z][a-zA-Z0-9]+", name)


def check_stimulus_naming_convention(name):
    assert re.match(r"[a-zA-Z0-9]+_?(?!0)\d+\.(?:jpg|jpeg|png|mp4)|(?!0)\d+\.(?:jpg|jpeg|png|mp4)", name)


def check_image_format(image, identifier):
    assert image.mode in ['RGBA', 'RGB', 'LA', 'L'], f"{identifier}: incorrect image mode {image.mode}"
    image_shape = np.array(image).shape
    assert 3 == len(image_shape), f"{identifier}: incorrect shape {len(image_shape)}"

    channels = {'RGBA': 4, 'RGB': 3, 'LA': 2, 'L': 1}
    assert channels[image.mode] == image_shape[2], f"{identifier}: incorrect channels {image_shape[2]}"


def check_stimulus_numbers(stimulus_set):
    stimulus_numbers = [int(stimulus_file_path[stimulus_file_path.rfind('_') + 1:stimulus_file_path.rfind('.')])
                     for stimulus_file_path in list(stimulus_set.stimulus_paths.values())]
    stimulus_numbers.sort()
    for i in range(len(stimulus_numbers) - 1):
        assert stimulus_numbers[i] == stimulus_numbers[i + 1] - 1, "StimulusSet files not sequentially numbered"


def check_experiment_stimulus_set(stimulus_set):
    """
    Checks the stimulus set files are non-corrupt and named/numbered sequentially. This function should only be called
    on stimulus sets that are pushed to the `brainio.requested` bucket.
    :param stimulus_set: A StimulusSet containing one row for each stimulus, and the columns
    {'stimulus_id', ['stimulus_path_within_store' (optional to structure zip directory layout)]}
    and columns for all stimulus-set-specific metadata but not the column 'filename'.
    """
    col_name = 'stimulus_id'
    if 'stimulus_id' not in stimulus_set.columns:
        col_name = 'image_id' # for legacy packages
    assert len(stimulus_set[col_name]), "StimulusSet is empty"
    file_paths = list(stimulus_set.stimulus_paths.values())

    file_type_0 = mimetypes.guess_type(file_paths[0])[0]

    for file_path in file_paths:
        check_stimulus_naming_convention(file_path[file_path.rfind('/') + 1:])
        assert os.path.isfile(file_path), f"{file_path} does not exist"
        assert file_type_0 == mimetypes.guess_type(file_path)[0], f"{file_path} is a different media type than other stimuli in the StimulusSet"
        if file_type_0.startswith('image'):
            image = Image.open(file_path)
            check_image_format(image, file_path)

    check_stimulus_numbers(stimulus_set)


def package_stimulus_set(catalog_name, proto_stimulus_set, stimulus_set_identifier, bucket_name="brainio-temp"):
    """
    Package a set of stimuli along with their metadata for the BrainIO system.
    :param catalog_name: The name of the lookup catalog to add the stimulus set to.
    :param proto_stimulus_set: A StimulusSet containing one row for each stimulus,
        and the columns {'stimulus_id', ['stimulus_path_within_store' (optional to structure zip directory layout)]}
        and columns for all stimulus-set-specific metadata but not the column 'filename'.
    :param stimulus_set_identifier: A unique name identifying the stimulus set
        <lab identifier>.<first author e.g. 'Rajalingham' or 'MajajHong' for shared first-author><YYYY year of publication>.
    :param bucket_name: The name of the bucket to upload to.
    """
    _logger.debug(f"Packaging {stimulus_set_identifier}")

    # for legacy packages
    id_col_present = 'stimulus_id' in proto_stimulus_set.columns or 'image_id' in proto_stimulus_set.columns
    assert id_col_present, "StimulusSet needs to have a `stimulus_id` column"

    if bucket_name == 'brainio.requested':
        check_experiment_stimulus_set(proto_stimulus_set)

    # naming
    stimulus_store_identifier = "stimulus_" + stimulus_set_identifier.replace(".", "_")
    # - csv
    csv_file_name = stimulus_store_identifier + ".csv"
    target_csv_path = Path(fetch.get_local_data_path()) / stimulus_store_identifier / csv_file_name
    # - zip
    zip_file_name = stimulus_store_identifier + ".zip"
    target_zip_path = Path(fetch.get_local_data_path()) / stimulus_store_identifier / zip_file_name
    # create csv and zip files
    stimulus_zip_sha1, zip_filenames = create_stimulus_zip(proto_stimulus_set, str(target_zip_path))
    assert 'filename' not in proto_stimulus_set.columns, "StimulusSet already has column 'filename'"
    proto_stimulus_set['filename'] = zip_filenames  # keep record of zip (or later local) filenames
    csv_sha1 = create_stimulus_csv(proto_stimulus_set, str(target_csv_path))
    # upload both to S3
    upload_to_s3(str(target_csv_path), bucket_name, target_s3_key=csv_file_name)
    upload_to_s3(str(target_zip_path), bucket_name, target_s3_key=zip_file_name)
    # link to csv and zip from same identifier. The csv however is the only one of the two rows with a class.
    lookup.append(
        catalog_identifier=catalog_name,
        object_identifier=stimulus_set_identifier, cls='StimulusSet',
        lookup_type=TYPE_STIMULUS_SET,
        bucket_name=bucket_name, sha1=csv_sha1, s3_key=csv_file_name,
        stimulus_set_identifier=None
    )
    lookup.append(
        catalog_identifier=catalog_name,
        object_identifier=stimulus_set_identifier, cls=None,
        lookup_type=TYPE_STIMULUS_SET,
        bucket_name=bucket_name, sha1=stimulus_zip_sha1, s3_key=zip_file_name,
        stimulus_set_identifier=None
    )
    _logger.debug(f"stimulus set {stimulus_set_identifier} packaged")


def write_netcdf(assembly, target_netcdf_file, append=False, group=None, compress=True):
    """
    Write a DataAssembly object to a netCDF file.
    :param assembly: The DataAssembly to write to the file.  DataAssembly or a subclass.
    :param target_netcdf_file: The file to write to.  str or path-like.
    :param append:  If true, add to an existing file instead of creating a new one.
    :param group:  If provided, the name of the netCDF group to write to within the file.  Otherwise the root group is used.  str.
    :param compress:  If true, write as compressed data.
    :return:  The SHA-1 hash of the file.  str.
    """
    assembly = assembly.copy()
    target_netcdf_file = Path(target_netcdf_file)
    _logger.debug(f"Writing assembly to {target_netcdf_file}")
    # reset_index can be removed when xarray supports writing MultiIndex to netCDF.
    assembly = assembly.reset_index(list(assembly.indexes))
    for name in list(assembly.attrs):
        attr = assembly.attrs[name]
        # We can't serialize complex objects to netCDF.
        # The following matches the complex objects that we tend to add as attributes.
        if isinstance(attr, pd.DataFrame) or isinstance(attr, xr.DataArray) or attr is None:
            del assembly.attrs[name]
    mode = "a" if append else "w"
    target_netcdf_file.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        ds = assembly.to_dataset(name="data")
        compression = dict(zlib=True, complevel=1)
        encoding = {var: compression for var in ds.variables}
        ds.to_netcdf(target_netcdf_file, mode=mode, group=group, encoding=encoding)
    else:
        assembly.to_netcdf(target_netcdf_file, mode=mode, group=group)
    sha1 = sha1_hash(target_netcdf_file)
    return sha1


def package_data_assembly(catalog_identifier, proto_data_assembly, assembly_identifier, stimulus_set_identifier,
                          assembly_class_name="NeuronRecordingAssembly", bucket_name="brainio-temp", extras=None):
    """
    Package a set of data along with its metadata for the BrainIO system.
    :param catalog_identifier: The name of the lookup catalog to add the data assembly to.
    :param proto_data_assembly: An xarray DataArray containing experimental measurements and all related metadata.
        * The dimensions of the DataArray must be appropriate for the DataAssembly class:
            * NeuroidAssembly and its subclasses:  "presentation", "neuroid"[, "time_bin"]
                * except for SpikeTimesAssembly:  "event"
            * MetaDataAssembly:  "event"
            * BehavioralAssembly:  should have a "presentation" dimension, but can be flexible about its other dimensions.
        * A presentation dimension must have a stimulus_id coordinate and should have coordinates for presentation-level metadata such as repetition.
          The presentation dimension should not have coordinates for stimulus-specific metadata, these will be drawn from the StimulusSet based on stimulus_id.
        * The neuroid dimension must have a neuroid_id coordinate and should have coordinates for as much neural metadata as possible (e.g. region, subregion, animal, row in array, column in array, etc.)
        * The time_bin dimension should have coordinates time_bin_start and time_bin_end.
    :param assembly_identifier: A dot-separated string starting with a lab identifier.
        * For published: <lab identifier>.<first author e.g. 'Rajalingham' or 'MajajHong' for shared first-author><YYYY year of publication>
        * For requests: <lab identifier>.<b for behavioral|n for neuroidal>.<m for monkey|h for human>.<proposer e.g. 'Margalit'>.<pull request number>
    :param stimulus_set_identifier: The unique name of an existing StimulusSet in the BrainIO system.
    :param assembly_class_name: The name of a DataAssembly subclass.
    :param bucket_name: The name of the bucket to upload to.
    """
    _logger.debug(f"Packaging {assembly_identifier}")

    # verify
    assembly_class = resolve_assembly_class(assembly_class_name)
    assembly = assembly_class(proto_data_assembly)
    assembly.attrs['stimulus_set_identifier'] = stimulus_set_identifier
    assembly.validate()
    assert stimulus_set_identifier in list_stimulus_sets(), \
        f"StimulusSet {stimulus_set_identifier} not found in packaged stimulus sets"

    # identifiers
    assembly_store_identifier = "assy_" + assembly_identifier.replace(".", "_")
    netcdf_file_name = assembly_store_identifier + ".nc"
    target_netcdf_path = Path(fetch.get_local_data_path()) / assembly_store_identifier / netcdf_file_name
    s3_key = netcdf_file_name

    # execute
    netcdf_kf_sha1 = write_netcdf(assembly, target_netcdf_path)
    if extras is not None:
        for k, ex in extras.items():
            assert isinstance(ex, DataArray)
            netcdf_kf_sha1 = write_netcdf(ex, target_netcdf_path, append=True, group=k)
    upload_to_s3(target_netcdf_path, bucket_name, s3_key)
    lookup.append(
        catalog_identifier=catalog_identifier,
        object_identifier=assembly_identifier, stimulus_set_identifier=stimulus_set_identifier,
        lookup_type=TYPE_ASSEMBLY,
        bucket_name=bucket_name, sha1=netcdf_kf_sha1,
        s3_key=s3_key, cls=assembly_class_name
    )
    _logger.debug(f"assembly {assembly_identifier} packaged")
