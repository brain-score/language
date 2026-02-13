from brainscore_language import data_registry
from brainscore_language.utils.s3 import load_from_s3

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

data_registry["Tuckute2024.language"] = lambda: load_from_s3(
    identifier="Tuckute2024.language",
    version_id="BB.DbwqLB4OhDR64duqojNdL0CRd4RmG",
    sha1="5c8fc7f3e24cc1af5f5296459377b638b6492641")