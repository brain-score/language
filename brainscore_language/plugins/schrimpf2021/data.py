import functools
import logging

from brainscore_language import datasets
from brainscore_language.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX_PEREIRA2018 = """@article{pereira2018toward,
  title={Toward a universal decoder of linguistic meaning from brain activation},
  author={Pereira, Francisco and Lou, Bin and Pritchett, Brianna and Ritter, Samuel and Gershman, Samuel J 
          and Kanwisher, Nancy and Botvinick, Matthew and Fedorenko, Evelina},
  journal={Nature communications},
  volume={9},
  number={1},
  pages={1--13},
  year={2018},
  publisher={Nature Publishing Group}
}"""

BIBTEX_FEDORENKO2016 = """@article{fedorenko2016neural,
  title={Neural correlate of the construction of sentence meaning},
  author={Fedorenko, Evelina and Scott, Terri L and Brunner, Peter and Coon, William G and Pritchett, Brianna and 
          Schalk, Gerwin and Kanwisher, Nancy},
  journal={Proceedings of the National Academy of Sciences},
  volume={113},
  number={41},
  pages={E6256--E6262},
  year={2016},
  publisher={National Acad Sciences}
}"""

BIBTEX_BLANK2014 = """@article{blank2014functional,
  title={A functional dissociation between language and multiple-demand systems revealed in patterns of BOLD signal fluctuations},
  author={Blank, Idan and Kanwisher, Nancy and Fedorenko, Evelina},
  journal={Journal of neurophysiology},
  volume={112},
  number={5},
  pages={1105--1118},
  year={2014},
  publisher={American Physiological Society Bethesda, MD}
}"""

datasets['Pereira2018.language_system'] = functools.partial(
    load_from_s3, identifier="Pereira2018.language_system", sha1="8b5056b79129e27cdbc2b9ee032e7ba06396f0f0")
datasets['Fedorenko2016.language'] = functools.partial(
    load_from_s3, identifier="Fedorenko2016.language", sha1="05ac632ee6cf4eaf0d71deb77afb688e7a965f9e")
datasets['Blank2014.fROI'] = functools.partial(
    load_from_s3, identifier="Blank2014.fROI", sha1="82c376712c26888fb769d02d3eca740e5f3a7679")
