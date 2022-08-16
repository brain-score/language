import logging

from brainio.assemblies import AssemblyLoader, BehavioralAssembly
from brainio.fetch import fetch_file
from brainscore_language import datasets

_logger = logging.getLogger(__name__)

BIBTEX = """@proceedings{futrell2018natural,
  title={The Natural Stories Corpus},
  author={Futrell, Richard and Gibson, Edward and Tily, Harry J. and Blank, Idan and Vishnevetsky, Anastasia and
          Piantadosi, Steven T. and Fedorenko, Evelina},
  conference={International Conference on Language Resources and Evaluation (LREC)},
  url={http://www.lrec-conf.org/proceedings/lrec2018/pdf/337.pdf},
  year={2018}
}"""


def load_from_s3():
    file_path = fetch_file(
        location_type="S3",
        location="https://brainscore-language.s3.amazonaws.com/assy_Futrell2018.nc",
        sha1="381ccc8038fbdb31235b5f3e1d350f359b5e287f",
    )
    loader = AssemblyLoader(cls=BehavioralAssembly, file_path=file_path)
    assembly = loader.load()
    assembly.attrs["identifier"] = "Futrell2018"
    return assembly


def load_from_local():
    file_path = ""
    loader = AssemblyLoader(cls=BehavioralAssembly, file_path=file_path)
    assembly = loader.load()
    assembly.attrs["identifier"] = "Futrell2018"
    return assembly


datasets["Futrell2018"] = load_from_s3
