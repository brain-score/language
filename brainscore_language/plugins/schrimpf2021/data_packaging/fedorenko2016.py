from glob import glob

import logging
import numpy as np
import os
import scipy.io
import scipy.stats
from pathlib import Path

from brainio.assemblies import NeuroidAssembly

_logger = logging.getLogger(__name__)


# adapted from
# https://github.com/mschrimpf/neural-nlp/blob/cedac1f868c8081ce6754ef0c13895ce8bc32efc/neural_nlp/neural_data/ecog.py#L18

def load_fedorenko2016():
    neural_data_dir = Path(__file__).parent / 'ecog-Fedorenko2016'
    stim_data_dir = Path(__file__).parent / 'sentences_8'
    _logger.info(f'Neural data directory: {neural_data_dir}')
    filepaths_stim = glob(os.path.join(stim_data_dir, '*.txt'))

    # Create a subject ID list corresponding to language electrodes
    subject1 = np.repeat(1, 47)
    subject2 = np.repeat(2, 8)
    subject3 = np.repeat(3, 9)
    subject4 = np.repeat(4, 15)
    subject5 = np.repeat(5, 18)

    filepath_neural = glob(os.path.join(neural_data_dir, '*g_lang_v3.mat'))

    _logger.debug(f'Running Fedorenko2016 benchmark with language responsive electrodes')

    ecog_mat = scipy.io.loadmat(filepath_neural[0])
    ecog_mtrix = ecog_mat['ecog']
    ecog_mtrix_T = np.transpose(ecog_mtrix)

    num_words = list(range(np.shape(ecog_mtrix_T)[0]))
    new_sent_idx = num_words[::8]

    # Average across word representations
    sent_avg_ecog = []
    for i in new_sent_idx:
        eight_words = ecog_mtrix_T[i:i + 8, :]
        sent_avg = np.mean(eight_words, 0)
        sent_avg_ecog.append(sent_avg)

    # Stimuli
    for filepath in filepaths_stim:
        with open(filepath, 'r') as file1:
            f1 = file1.readlines()

        _logger.debug(f1)

        sentences = []
        sentence_words, word_nums = [], []
        for sentence in f1:
            sentence = sentence.split(' ')
            sentences.append(sentence)
            word_counter = 0

            for word in sentence:
                if word == '\n':
                    continue
                word = word.rstrip('\n')
                sentence_words.append(word)
                word_nums.append(word_counter)
                word_counter += 1

        _logger.debug(sentence_words)

    # Create sentenceID list
    sentence_lst = list(range(0, 52))
    sentenceID = np.repeat(sentence_lst, 8)

    subjectID = np.concatenate([subject1, subject2, subject3, subject4, subject5], axis=0)

    # Create a list for each word number
    word_number = list(range(np.shape(ecog_mtrix_T)[0]))

    # xarray
    electrode_numbers = list(range(np.shape(ecog_mtrix_T)[1]))
    assembly = NeuroidAssembly(ecog_mtrix_T,
                               dims=('presentation', 'neuroid'),
                               coords={'stimulus_id': ('presentation', word_number),
                                       'word': ('presentation', sentence_words),
                                       'word_num': ('presentation', word_nums),
                                       'sentence_id': ('presentation', sentenceID),
                                       'electrode': ('neuroid', electrode_numbers),
                                       'neuroid_id': ('neuroid', electrode_numbers),
                                       'subject_UID': ('neuroid', subjectID),  # Name is subject_UID for consistency
                                       })

    return assembly
