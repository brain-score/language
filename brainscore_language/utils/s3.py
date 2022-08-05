from brainio.assemblies import AssemblyLoader, NeuroidAssembly
from brainio.fetch import fetch_file


def load_from_s3(identifier, sha1) -> NeuroidAssembly:
    filename = f"assy_{identifier.replace('.', '_')}.nc"
    file_path = fetch_file(location_type="S3",
                           location=f"https://brainscore-language.s3.amazonaws.com/{filename}",
                           sha1=sha1)
    loader = AssemblyLoader(cls=NeuroidAssembly, file_path=file_path)
    assembly = loader.load()
    assembly.attrs['identifier'] = identifier
    return assembly
