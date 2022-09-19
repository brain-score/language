import logging
import numpy as np
import pandas as pd
import re
import sys
from pathlib import Path
from tqdm import tqdm

from brainscore_core import BehavioralAssembly
from brainscore_language.utils.s3 import upload_data_assembly

_logger = logging.getLogger(__name__)

"""
The code in this file was run only once to initially upload the data, and is kept here for reference.
"""


def upload_natural_stories():
    # adapted from
    # https://github.com/mschrimpf/neural-nlp/blob/cedac1f868c8081ce6754ef0c13895ce8bc32efc/neural_nlp/neural_data/naturalStories.py#L15
    data_path = Path(__file__).parent / 'naturalstories_RTS'
    data_file = data_path / 'processed_RTs.csv'
    stories_file = data_path / 'all_stories.tok'
    _logger.info(f'Data file: {data_file}. Stories file: {stories_file}.')

    # get data
    data = pd.read_csv(data_file)

    # get unique word identifier tuples and order in order of stories
    item_ID = np.array(data['item'])
    zone_ID = np.array(data['zone'])
    zpd_lst = list(zip(item_ID, zone_ID))
    unique_zpd_lst = list(set(zpd_lst))
    unique_zpd_lst = sorted(unique_zpd_lst, key=lambda tup: (tup[0], tup[1]))

    # get unique WorkerIds
    subjects = data.WorkerId.unique()

    # ====== create reading_times ======
    r_dim = len(unique_zpd_lst)
    c_dim = len(subjects)

    # default value for a subject's not having an RT for a story/word is NaN
    reading_times = np.empty((r_dim, c_dim))
    reading_times[:] = np.nan

    # set row and column indices for reading_times
    r_indices = {unique_zpd_lst[i]: i for i in range(r_dim)}
    c_indices = {subjects[i]: i for i in range(c_dim)}

    # populate meta information dictionary for subjects xarray dimension
    metaInfo_subjects = {}

    for index, d in tqdm(data.iterrows(), total=len(data), desc='indices'):
        r = r_indices[(d['item'], d['zone'])]
        c = c_indices[d['WorkerId']]
        reading_times[r][c] = d['RT']
        key = d['WorkerId']
        if key not in metaInfo_subjects:
            metaInfo_subjects[key] = (d['correct'], d['WorkTimeInSeconds'])

    reading_times = np.array(reading_times)

    # get subjects' metadata
    correct_meta = [v[0] for v in metaInfo_subjects.values()]
    WorkTimeInSeconds_meta = [v[1] for v in metaInfo_subjects.values()]

    # get metadata for presentation dimension
    word_df = pd.read_csv(stories_file, sep='\t')
    voc_item_ID = np.array(word_df['item'])
    voc_zone_ID = np.array(word_df['zone'])
    voc_word = np.array(word_df['word'])

    # get sentence_IDs (finds 481 sentences)
    sentence_ID = []
    idx = 1
    for i, elm in enumerate(voc_word):
        sentence_ID.append(idx)
        if elm.endswith((".", "?", "!", ".'", "?'", "!'", ";'")):
            if i + 1 < len(voc_word):
                if not (voc_word[i + 1].islower() or voc_word[i] == "Mr."):
                    idx += 1

    # get IDs of words within a sentence
    word_within_a_sentence_ID = []
    idx = 0
    for i, elm in enumerate(voc_word):
        idx += 1
        word_within_a_sentence_ID.append(idx)
        if elm.endswith((".", "?", "!", ".'", "?'", "!'", ";'")):
            if i + 1 < len(voc_word):
                if not (voc_word[i + 1].islower() or voc_word[i] == "Mr."):
                    idx = 0
            else:
                idx = 0

    # stimulus_ID
    stimulus_ID = list(range(1, len(voc_word) + 1))

    # add word_core that treats e.g. "\This" and "This" as the same words (to split over)
    word_core = [re.sub(r'[^\w\s]', '', word) for word in voc_word]

    # build xarray
    # voc_word = word
    # voc_item_ID = index of story (1-10)
    # voc_zone_ID = index of words within a story
    # sentence_ID = index of words within each sentence
    # stimulus_ID = unique index of word across all stories
    # subjects = WorkerIDs
    # correct_meta = number of correct answers in comprehension questions
    assembly = BehavioralAssembly(reading_times,
                                  dims=('presentation', 'subject'),
                                  coords={'word': ('presentation', voc_word),
                                          'word_core': ('presentation', word_core),
                                          'story_id': ('presentation', voc_item_ID),
                                          'word_id': ('presentation', voc_zone_ID),
                                          'word_within_sentence_id': ('presentation', word_within_a_sentence_ID),
                                          'sentence_id': ('presentation', sentence_ID),
                                          'stimulus_id': ('presentation', stimulus_ID),
                                          'subject_id': ('subject', subjects),
                                          'correct': ('subject', correct_meta),
                                          'WorkTimeInSeconds': ('subject', WorkTimeInSeconds_meta),
                                          })

    # upload
    upload_data_assembly(assembly,
                         assembly_identifier="Futrell2018")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    upload_natural_stories()
