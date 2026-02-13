import logging
from pathlib import Path

from brainscore_core.supported_data_standards.brainio import fetch
from brainscore_core.supported_data_standards.brainio.assemblies import AssemblyLoader, NeuroidAssembly, DataAssembly
from brainscore_core.supported_data_standards.brainio.fetch import fetch_file
from brainscore_core.supported_data_standards.brainio.packaging import write_netcdf, upload_to_s3

_logger = logging.getLogger(__name__)


def upload_data_assembly(assembly, assembly_identifier, bucket_name="brainscore-language", assembly_prefix="assy_"):
    # adapted from
    # https://github.com/mschrimpf/brainio/blob/8a40a3558d0b86072b9e221808f19005c7cb8c17/brainio/packaging.py#L217

    _logger.debug(f"Uploading {assembly_identifier} to S3")

    # identifiers
    assembly_store_identifier = assembly_prefix + assembly_identifier.replace(".", "_")
    netcdf_file_name = assembly_store_identifier + ".nc"
    target_netcdf_path = Path(fetch.get_local_data_path()) / assembly_store_identifier / netcdf_file_name
    s3_key = netcdf_file_name

    # write to disk and upload
    netcdf_kf_sha1 = write_netcdf(assembly, target_netcdf_path)
    response = upload_to_s3(target_netcdf_path, bucket_name, s3_key)
    _logger.debug(f"Uploaded {assembly_store_identifier} to S3 "
                  f"with key={s3_key}, sha1={netcdf_kf_sha1}, version_id={response['VersionId']}: {response}")
    response['sha1'] = netcdf_kf_sha1
    return response


def load_from_s3(identifier, version_id, sha1, assembly_prefix="assy_", cls=NeuroidAssembly) -> DataAssembly:
    filename = f"{assembly_prefix}{identifier.replace('.', '_')}.nc"
    file_path = fetch_file(location_type="S3",
                           location=f"https://brainscore-language.s3.amazonaws.com/{filename}",
                           version_id=version_id,
                           sha1=sha1)
    loader = AssemblyLoader(cls=cls, file_path=file_path)
    assembly = loader.load()
    assembly.attrs['identifier'] = identifier
    return assembly
