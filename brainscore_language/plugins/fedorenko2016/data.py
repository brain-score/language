import logging

from brainscore_language import datasets
from brainscore_language.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """@article{fedorenko2016neural,
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
datasets['Fedorenko2016.language'] = lambda: load_from_s3(
    identifier="Fedorenko2016.language",
    version_id="qvB7YZfEjbXEE64bODNLlQlZKWGpgPhy",
    sha1="2966b6d78e972a72068aa6907377483f427e8d9a")
