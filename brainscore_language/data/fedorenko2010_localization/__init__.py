import pandas as pd

from glob import glob
from pathlib import Path

from brainscore_language import data_registry

BIBTEX = """@article{Fedorenko2010NewMF,
  title={New method for fMRI investigations of language: defining ROIs functionally in individual subjects.},
  author={Evelina Fedorenko and Po-Jang Hsieh and Alfonso Nieto-Castanon and Susan L. Whitfield-Gabrieli and Nancy G. Kanwisher},
  journal={Journal of neurophysiology},
  year={2010},
  volume={104 2},
  pages={1177-94},
  url={https://api.semanticscholar.org/CorpusID:740913}
}"""

# Code adapted from: https://github.com/bkhmsi/brain-language-suma

def load_data():
    paths = glob(f"{Path(__file__).parent }/*.csv")
    data = pd.read_csv(paths[0])
    for path in paths[1:]:
        run_data = pd.read_csv(path)
        data = pd.concat([data, run_data])

    data["sent"] = data["stim2"].apply(str.lower)

    for stimuli_idx in range(3, 14):
        data["sent"] += " " + data[f"stim{stimuli_idx}"].apply(str.lower)
    return data

data_registry['Fedorenko2010.localization'] = load_data