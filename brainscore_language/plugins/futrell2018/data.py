import logging

from brainscore_language import datasets
from brainscore_language.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """@proceedings{futrell2018natural,
  title={The Natural Stories Corpus},
  author={Futrell, Richard and Gibson, Edward and Tily, Harry J. and Blank, Idan and Vishnevetsky, Anastasia and
          Piantadosi, Steven T. and Fedorenko, Evelina},
  conference={International Conference on Language Resources and Evaluation (LREC)},
  url={http://www.lrec-conf.org/proceedings/lrec2018/pdf/337.pdf},
  year={2018}
}"""

datasets['Futrell2018'] = lambda: load_from_s3(
    identifier="Futrell2018", sha1="381ccc8038fbdb31235b5f3e1d350f359b5e287f")
