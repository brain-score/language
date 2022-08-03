import logging
import sys
from pathlib import Path

from brainio import fetch
from brainio.packaging import write_netcdf, upload_to_s3
from brainscore_language.plugins.schrimpf2021.data_packaging.blank2014 import load_blank2014
from brainscore_language.plugins.schrimpf2021.data_packaging.fedorenko2016 import load_fedorenko2016
from brainscore_language.plugins.schrimpf2021.data_packaging.pereira2018 import load_pereira2018

_logger = logging.getLogger(__name__)

"""
The code in this package was run only once to initially upload the data, and is kept here for reference.
"""


def upload_pereira2018():
    assembly = load_pereira2018()
    upload_data_assembly(assembly,
                         assembly_identifier="Pereira2018.language_system",
                         bucket_name="brainscore-language")


def upload_fedorenko2016():
    assembly = load_fedorenko2016()
    upload_data_assembly(assembly,
                         assembly_identifier="Fedorenko2016.language",
                         bucket_name="brainscore-language")


def upload_blank2014():
    assembly = load_blank2014()
    upload_data_assembly(assembly,
                         assembly_identifier="Blank2014.fROI4s",
                         bucket_name="brainscore-language")


def _build_id(assembly, coords):
    return [".".join([f"{value}" for value in values]) for values in zip(*[assembly[coord].values for coord in coords])]


def upload_data_assembly(assembly, assembly_identifier, bucket_name):
    # adapted from
    # https://github.com/mschrimpf/brainio/blob/8a40a3558d0b86072b9e221808f19005c7cb8c17/brainio/packaging.py#L217

    _logger.debug(f"Uploading {assembly_identifier} to S3")

    # identifiers
    assembly_store_identifier = "assy_" + assembly_identifier.replace(".", "_")
    netcdf_file_name = assembly_store_identifier + ".nc"
    target_netcdf_path = Path(fetch.get_local_data_path()) / assembly_store_identifier / netcdf_file_name
    s3_key = netcdf_file_name

    # write to disk and upload
    netcdf_kf_sha1 = write_netcdf(assembly, target_netcdf_path)
    upload_to_s3(target_netcdf_path, bucket_name, s3_key)
    _logger.debug(f"Uploaded assembly {assembly_identifier} to S3: {s3_key} (SHA1 hash {netcdf_kf_sha1})")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    upload_pereira2018()
    upload_fedorenko2016()
    upload_blank2014()
