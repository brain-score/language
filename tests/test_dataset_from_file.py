
from pathlib import Path

import pandas as pd
import xarray as xr

from langbrainscore.utils.logging import log
from langbrainscore.dataset import Dataset


def test_load_pereira_data():

    # define the original Pereira data file
    filepath = Path(Path(__file__).parents[1] / 'data/Pereira_FirstSession_TrialEffectSizes_20220223.csv').resolve()

    # first, get the pereira data file in the format expected by our Dataset
    df = pd.read_csv(filepath)
    id_columns = df.columns[:9]
    neuroid_columns = df.columns[9:]
    df = df.melt(id_vars=id_columns, value_name='effect_size', value_vars=neuroid_columns)
    df.rename(columns={'variable': 'neuroid'}, inplace=True)
    df.to_csv(Path(__file__).parents[1] / 'data/molten_Pereira_FirstSession_TrialEffectSizes_20220223.csv')

    # next, use this transformed data to construct the dataset
    filepath = filepath.parent / ('molten_' + filepath.parts[-1])

    log(f'constructing a Dataset using {filepath}')

    dataset = Dataset.from_file_or_url(filepath, 
                                       data_column='effect_size',
                                       sampleid_index='Stim', 
                                       neuroid_index='neuroid',
                                       subject_index='UID',
                                       
                                       sampleid_metadata={'Stim': 'stim_identifier',
                                                          'Sentence': 'sentence',
                                                         },
                                       neuroid_metadata={'neuroid': 'roi',
                                                         'Session': 'session',
                                                         'DurationTR': 'tr',
                                                         'Experiment': 'experiment',
                                                         'UID': 'subject'},
                                       timeid_metadata=None,

                                       multidim_metadata=None, # if we had, 
                                                               # for example, a stimulus unfolding with time
                                       sort_by=['UID', 'Session', 'Experiment'],
                                    )

    return False


if __name__ == '__main__':
    test_load_pereira_data()