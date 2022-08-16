import logging

from brainio.assemblies import AssemblyLoader, BehavioralAssembly
from brainio.fetch import fetch_file
from brainscore_language import datasets

_logger = logging.getLogger(__name__)

BIBTEX = """@article{Pereira2018TowardAU,
  title={Toward a universal decoder of linguistic meaning from brain activation},
  author={Francisco Pereira and Bin Lou and Brianna Pritchett and Samuel Ritter and Samuel J. Gershman and Nancy G. Kanwisher and Matthew M. Botvinick and Evelina Fedorenko},
  journal={Nature Communications},
  year={2018},
  volume={9}
}"""


def load_from_s3():
    file_path = fetch_file(
        location_type="S3",
        location="https://brainscore-language.s3.amazonaws.com/assy_Futrell2018.nc",  # todo update paths
        sha1="381ccc8038fbdb31235b5f3e1d350f359b5e287f",
    )
    loader = AssemblyLoader(cls=BehavioralAssembly, file_path=file_path)
    assembly = loader.load()
    assembly.attrs["identifier"] = "Pereira2018ROI"
    return assembly


def load_from_local():
    from pathlib import Path

    file_path = Path(__file__).parent / "Pereira2018_Lang_fROI.csv"
    loader = AssemblyLoader(cls=BehavioralAssembly, file_path=file_path)
    assembly = loader.load()
    assembly.attrs["identifier"] = "Pereira2018ROI"
    return assembly


datasets["Pereira2018ROI"] = load_from_local
