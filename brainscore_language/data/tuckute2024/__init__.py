import logging

from brainscore_language import data_registry
from brainscore_language.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """@article{article,
        author = {Tuckute, Greta and Sathe, Aalok and Srikant, Shashank and Taliaferro, Maya and Wang, Mingye and Schrimpf, Martin and Kay, Kendrick and Fedorenko, Evelina},
        year = {2024},
        month = {01},
        pages = {1-18},
        title = {Driving and suppressing the human language network using large language models},
        volume = {8},
        journal = {Nature Human Behaviour},
        doi = {10.1038/s41562-023-01783-7}
        }"""

# TODO
# data_registry['Tuckute2024.5subj.lang_LH_netw'] = lambda: load_from_s3(
#     identifier="Pereira2018.language",
#     version_id="fq0gh.P7ThLu6DWUulho5W_F.YTEhDqJ",
#     sha1="f8434b4022f5b2c862f0ff2854d5b3f5f2a7fb96")

