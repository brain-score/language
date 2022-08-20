import logging

from brainscore_language import datasets
from brainscore_language.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """@article{blank2014functional,
  title={A functional dissociation between language and multiple-demand systems revealed in patterns of BOLD signal fluctuations},
  author={Blank, Idan and Kanwisher, Nancy and Fedorenko, Evelina},
  journal={Journal of neurophysiology},
  volume={112},
  number={5},
  pages={1105--1118},
  year={2014},
  publisher={American Physiological Society Bethesda, MD}
}"""

datasets['Blank2014.fROI'] = lambda: load_from_s3(
    identifier="Blank2014.fROI",
    version_id="qM.uLV8ltOHM297r2SaGteYMX4Vy.oHB",
    sha1="af1e868821b897cb1684e4c8dcd33977121ef552")
