import logging

from brainscore_language import data_registry
from brainscore_language.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """@article{pereira2018toward,
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

data_registry["Pereira2018.language"] = lambda: load_from_s3(
    identifier="Pereira2018.language",
    version_id="fq0gh.P7ThLu6DWUulho5W_F.YTEhDqJ",
    sha1="f8434b4022f5b2c862f0ff2854d5b3f5f2a7fb96",
    assembly_prefix="assy_",
)
data_registry["Pereira2018.auditory"] = lambda: load_from_s3(
    identifier="Pereira2018.auditory",
    version_id=".lCMuSrGBlsEgLtZDOApLlr3h2szCmoC",
    sha1="08e576bd3b8caf64850bb879abf07ae228ff1f5f",
    assembly_prefix="assy_",
)
