import logging
from pathlib import Path

from brainio import fetch
from brainio.assemblies import AssemblyLoader, NeuroidAssembly, DataAssembly


def load_from_disk(
    identifier,
    assembly_prefix="assembly_",
    cls=NeuroidAssembly,
) -> DataAssembly:

    assembly_store_identifier = f"{assembly_prefix}" + identifier.replace(".", "_")
    # identifiers
    netcdf_file_name = assembly_store_identifier + ".nc"
    target_netcdf_path = (
        Path(fetch.get_local_data_path()) / assembly_store_identifier / netcdf_file_name
    )

    loader = AssemblyLoader(
        cls=cls,
        file_path=target_netcdf_path,
    )
    assembly = loader.load()
    assembly.attrs["identifier"] = identifier
    return assembly
