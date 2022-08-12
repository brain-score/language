from collections import namedtuple, defaultdict

import logging
import numpy as np
import operator
import pandas as pd
import scipy.io
import scipy.io
import scipy.stats
import scipy.stats
import sys
import warnings
from nltk_contrib.textgrid import TextGrid
from pathlib import Path
from tqdm import tqdm

from brainio.assemblies import walk_coords, merge_data_arrays, array_is_element, DataAssembly
from brainio.stimuli import StimulusSet
from brainscore_language.utils.s3 import upload_data_assembly

_logger = logging.getLogger(__name__)

"""
The code in this package was run only once to initially upload the data, and is kept here for reference.
"""


def upload_blank2014():
    assembly = load_blank2014()
    upload_data_assembly(assembly,
                         assembly_identifier="Blank2014.fROI",
                         bucket_name="brainscore-language")


# This file requires nltk_contrib to be installed to run. nltk_contrib is not part of requirements.txt because this
# script is only run once for data upload.

# adapted from
# https://github.com/mschrimpf/neural-nlp/blob/cedac1f868c8081ce6754ef0c13895ce8bc32efc/neural_nlp/neural_data/fmri.py#L241


def load_blank2014():
    bold_shift = 4
    assembly = load_voxels(bold_shift_seconds=bold_shift)
    assembly = average_subregions(assembly)
    return assembly


def load_voxels(bold_shift_seconds=4):
    assembly = load_voxel_data(bold_shift_seconds=bold_shift_seconds)
    assembly['subject_id'] = assembly['subject_UID']
    assembly = DataAssembly(assembly)
    stimulus_set = NaturalisticStories()()
    stimulus_set, assembly = _align_stimuli_recordings(stimulus_set, assembly)
    assert set(assembly['stimulus_sentence'].values).issubset(set(stimulus_set['sentence']))
    return assembly


def load_voxel_data(bold_shift_seconds=4):
    data = load_filtered_voxel_timepoints()
    data = DataAssembly(data)
    meta = load_time_meta()
    annotated_data = _merge_voxel_meta(data, meta, bold_shift_seconds)
    return annotated_data


def load_filtered_voxel_timepoints():
    data = load_voxel_timepoints()
    data = DataAssembly(data)
    data = data.sel(threshold='from90to100')
    data = data.sel(subject_nStories=8)
    return data


stories_meta = ['Boar', 'Aqua', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'MrSticky',
                'HighSchool', 'Roswell', 'Tulips', 'Tourette', 'Boar']
stories_meta = DataAssembly(stories_meta, coords={
    'story_name': ('story', stories_meta),
    'number': ('story', list(range(1, 11)) + [1]),
    'reader': ('story', ['Ted', 'Ted', 'Nancy', 'Nancy', 'Ted', 'Nancy', 'Nancy', 'Ted', 'Ted', 'Nancy', 'Terri']),
    'time_with_fixation': ('story', [338, 318, 394, 396, 302, 410, 348, 394, 388, 422, 338]),
    'time_without_fixation': ('story', [5 * 60 + 6, 4 * 60 + 46, 6 * 60 + 2, 6 * 60 + 4, 4 * 60 + 30,
                                        6 * 60 + 18, 5 * 60 + 16, 6 * 60 + 2, 5 * 60 + 56, 6 * 60 + 30, 5 * 60 + 6]),
    'recording_timepoints': ('story', [169, 159, 197, 198, 151, 205, 174, 197, 194, 211, 169])
}, dims=['story'])
stories_meta['story_index'] = 'story', [".".join([str(value) for value in values]) for values in zip(*[
    stories_meta[coord].values for coord in ['story_name', 'reader']])]

fROIs = {
    'language': [
        '01_LH_PostTemp',
        '02_LH_AntTemp',
        '03_LH_IFG',
        '04_LH_IFGorb',
        '05_LH_MFG',
        '06_LH_AngG',
        '07_RH_PostTemp',
        '08_RH_AntTemp',
        '09_RH_IFG',
        '10_RH_IFGorb',
        '11_RH_MFG',
        '12_RH_AngG',
    ],
    'MD_langloc': [
        '01_LH_postParietal',
        '02_LH_midParietal',
        '03_LH_antParietal',
        '04_LH_supFrontal',
        '05_LH_Precentral_A_PrecG',
        '06_LH_Precental_B_IFGop',
        '07_LH_midFrontal',
        '08_LH_midFrontalOrb',
        '09_LH_insula',
        '10_LH_medialFrontal',
        '11_RH_postParietal',
        '12_RH_midParietal',
        '13_RH_antParietal',
        '14_RH_supFrontal',
        '15_RH_Precentral_A_PrecG',
        '16_RH_Precental_B_IFGop',
        '17_RH_midFrontal',
        '18_RH_midFrontalOrb',
        '19_RH_insula',
        '20_RH_medialFrontal',
    ],
    'DMN_langloc': [
        '01_LH_FrontalMed.img',
        '02_LH_PostCing.img',
        '03_LH_TPJ.img',
        '04_LH_MidCing.img',
        '05_LH_STGorInsula.img',
        '06_LH_AntTemp.img',
        '07_RH_FrontalMed.img',
        '08_RH_PostCing.img',
        '09_RH_TPJ.img',
        '10_RH_MidCing.img',
        '11_RH_STGorInsula.img',
        '12_RH_AntTemp.img',
    ],
    'auditory': [
        '01_LH_TE11.img',
        '02_LH_TE12.img',
        '03_RH_TE11.img',
        '04_RH_TE12.img',
    ]
}
fROIs['MD_spatWM'] = fROIs['MD_langloc']
fROIs['DMN_spatWM'] = fROIs['DMN_langloc']


def load_voxel_timepoints():
    def _dim_coord_values(assembly):
        dim_coord_values = defaultdict(dict)
        for coord, dims, values in walk_coords(assembly):
            assert len(dims) == 1
            dim = dims[0]
            dim_coord_values[dim][coord] = values.tolist()
        return dim_coord_values

    def _dim_index(dim_coord_values):
        dim_values = {}
        for dim, coord_values in dim_coord_values.items():
            values = [dict(zip(coord_values, t)) for t in zip(*coord_values.values())]
            values = ["__".join(str(value) for value in row_dict.values()) for row_dict in values]
            dim_values[dim] = values
        return dim_values

    # 1st pass: find unique coords
    dim_coord_values = defaultdict(lambda: defaultdict(list))
    for story_data in _iterate_voxel_timepoints(desc='pass 1: coords'):
        story_dim_values = _dim_coord_values(story_data)
        for dim, dict_values in story_dim_values.items():
            for coord, values in dict_values.items():
                dim_coord_values[dim][coord] += values

    dim_index = _dim_index(dim_coord_values)
    dim_index = {dim: np.unique(values, return_index=True) for dim, values in dim_index.items()}
    coords = {coord: (dim, np.array(values)[dim_index[dim][1]])
              for dim, coord_values in dim_coord_values.items() for coord, values in coord_values.items()}

    # 2nd pass: fill coords with data
    data = np.empty([len(values) for (values, index) in dim_index.values()])
    data[:] = np.nan
    for story_data in _iterate_voxel_timepoints(desc='pass 2: data'):
        story_dim_index = _dim_index(_dim_coord_values(story_data))
        indices = {dim: np.searchsorted(dim_index[dim][0], story_dim_index[dim]) for dim in dim_index}
        indices = [indices[dim] for dim in story_data.dims]
        data[np.ix_(*indices)] = story_data.values
    data = DataAssembly(data, coords=coords, dims=['threshold', 'neuroid', 'timepoint'])
    data['neuroid_id'] = 'neuroid', [".".join([str(value) for value in values]) for values in zip(*[
        data[coord].values for coord in ['subject_UID', 'region', 'fROI_area', 'voxel_num']])]
    return data


def _iterate_voxel_timepoints(desc='files'):
    fixation_offset = 8
    data_dir = Path(__file__).parent / 'StoriesData_Dec2018'
    meta_filepath = data_dir / 'subjectsWithStoryData_andPreprocessedTimeseries_20190118.mat'
    meta = scipy.io.loadmat(meta_filepath)
    meta = meta['ssStruct']
    subject_meta = {key: [value.squeeze().tolist() for value in meta[key][0]] for key in list(meta.dtype.fields)}
    subject_meta['stories'] = [[story.squeeze().tolist() for story in stories] for stories in subject_meta['stories']]

    files = [data_dir / f"{uid}_{session_id}_preprocessed.mat" for uid, session_id in
             zip(subject_meta['UID'], subject_meta['SessionID'])]
    nonexistent_files = [file for file in files if not file.exists()]
    assert not nonexistent_files, f"Files {nonexistent_files} do not exist"
    file_subject_meta = [dict(zip(subject_meta, t)) for t in zip(*subject_meta.values())]
    for subject_meta, filepath in tqdm(zip(file_subject_meta, files), total=len(file_subject_meta), desc=desc):
        if subject_meta['UID'] not in ['088', '085', '098', '061', '090']:
            continue
        f = scipy.io.loadmat(filepath)
        file_data = f['data']
        regions = list(file_data.dtype.fields)
        for region in regions:
            if region not in ['language']:
                continue
            region_data = file_data[region][0, 0][0, 0]
            thresholds = list(region_data.dtype.fields)
            for threshold in thresholds:
                if threshold not in ['from90to100']:
                    continue
                threshold_data = region_data[threshold].squeeze()
                num_fROIs = threshold_data.shape[0]
                for fROI_index in range(num_fROIs):
                    fROI_name = fROIs[region][fROI_index]
                    timeseries = threshold_data[fROI_index]['timeseries'].squeeze()
                    if timeseries.dtype.fields is None:
                        assert np.isnan(timeseries)
                        warnings.warn(f"NaN timeseries: {filepath}, region {region}, threshold {threshold}, "
                                      f"fROI {fROI_name}/{fROI_index}")
                        continue
                    stories = list(timeseries.dtype.fields)
                    for story in stories:
                        story_data = timeseries[story].tolist()
                        subject_story_index = subject_meta['stories'].index(story)
                        num_neuroids, num_timepoints = story_data.shape[0], story_data.shape[1]

                        story_data = DataAssembly([story_data], coords={**{
                            'threshold': [threshold],
                            'voxel_num': ('neuroid', np.arange(0, num_neuroids)),
                            'region': ('neuroid', [region] * num_neuroids),
                            'fROI_area': ('neuroid', [fROI_name] * num_neuroids),
                            'fROI_index': ('neuroid', [fROI_index] * num_neuroids),
                            'timepoint_value': ('timepoint', np.arange(
                                2 - fixation_offset, 2 + num_timepoints * 2 - fixation_offset, 2)),  # 2s snapshots
                            'story': ('timepoint', [story] * num_timepoints),
                        }, **{
                            f"subject_{key}": ('neuroid', [value] * num_neuroids)
                            for key, value in subject_meta.items()
                            if key not in ['stories', 'storiesComprehensionScores', 'storiesComprehensionUnanswered']
                        }}, dims=['threshold', 'neuroid', 'timepoint'])
                        yield story_data


def load_time_meta():
    data_dir = Path(__file__).parent / 'StoriesData_Dec2018' / 'stories_textgridsbyJeanne'
    files = data_dir.glob("*TextGrid*")
    time_to_words = []
    for file in files:
        textgrid = TextGrid.load(file)
        words = [tier for tier in textgrid.tiers if tier.nameid == 'words'][0]
        rows = defaultdict(list)
        for (time_start, time_end, word) in words.simple_transcript:
            rows['time_start'].append(float(time_start))
            rows['time_end'].append(float(time_end))
            rows['word'].append(word)
        story_index = int(file.stem)
        story = stories_meta.sel(number=story_index).values
        story = next(iter(set(story)))  # Boar was read twice
        rows = DataAssembly(rows['word'],
                            coords={'filepath': ('time_bin', [file.name] * len(rows['word'])),
                                    'story': ('time_bin', [story] * len(rows['word'])),
                                    'time_start': ('time_bin', rows['time_start']),
                                    'time_end': ('time_bin', rows['time_end']),
                                    },
                            dims=['time_bin'])
        time_to_words.append(rows)
    time_to_words = merge_data_arrays(time_to_words)
    return time_to_words


def _merge_voxel_meta(data, meta, bold_shift_seconds):
    data_missing = set(meta['story'].values) - set(data['story'].values)
    if data_missing:
        warnings.warn(f"Stories missing from the data: {data_missing}")
    meta_missing = set(data['story'].values) - set(meta['story'].values)
    if meta_missing:
        warnings.warn(f"Stories missing from the meta: {meta_missing}")

    ignored_words = [None, '', '<s>', '</s>', '<s']
    annotated_data = []
    for story in tqdm(ordered_set(data['story'].values), desc='merge meta'):
        if story not in meta['story'].values:
            continue
        story_meta = meta.sel(story=story)
        story_meta = story_meta.sortby('time_end')

        story_data = data.sel(story=story).stack(timepoint=['timepoint_value'])
        story_data = story_data.sortby('timepoint_value')
        timepoints = story_data['timepoint_value'].values.tolist()
        assert is_sorted(timepoints)
        timepoints = [timepoint - bold_shift_seconds for timepoint in timepoints]
        sentences = []
        last_timepoint = -np.inf
        for timepoint in timepoints:
            if last_timepoint >= max(story_meta['time_end'].values):
                break
            if timepoint <= 0:
                sentences.append(None)
                continue  # ignore fixation period
            timebin_meta = [last_timepoint < end <= timepoint for end in story_meta['time_end'].values]
            timebin_meta = story_meta[{'time_bin': timebin_meta}]
            sentence = ' '.join(word.strip() for word in timebin_meta.values if word not in ignored_words)
            sentence = sentence.lower().strip()
            # quick-fixes
            if story == 'Boar' and sentence == 'interactions the the':  # Boar duplicate
                sentence = 'interactions the'
            if story == 'KingOfBirds' and sentence == 'the fact that the larger':  # missing word in TextGrid
                sentence = 'earth ' + sentence
            if story == 'MrSticky' and sentence == 'worry don\'t worry i went extra slowly since it\'s':
                sentence = 'don\'t worry i went extra slowly since it\'s'
            sentences.append(sentence)
            last_timepoint = timebin_meta['time_end'].values[-1]
        sentence_index = [i for i, sentence in enumerate(sentences) if sentence]
        sentences = np.array(sentences)[sentence_index]
        if story not in ['Boar', 'KingOfBirds', 'MrSticky']:  # ignore quick-fixes
            annotated_sentence = ' '.join(sentences)
            meta_sentence = ' '.join(word.strip() for word in story_meta.values if word not in ignored_words) \
                .lower().strip()
            assert annotated_sentence == meta_sentence
        # re-interpret timepoints as stimuli
        coords = {}
        for coord_name, dims, coord_value in walk_coords(story_data):
            dims = [dim if not dim.startswith('timepoint') else 'presentation' for dim in dims]
            # discard the timepoints for which the stimulus did not change (empty word)
            coord_value = coord_value if not array_is_element(dims, 'presentation') else coord_value[sentence_index]
            coords[coord_name] = dims, coord_value
        coords = {**coords, **{'stimulus_sentence': ('presentation', sentences)}}
        story_data = story_data[{dim: slice(None) if dim != 'timepoint' else sentence_index
                                 for dim in story_data.dims}]
        dims = [dim if not dim.startswith('timepoint') else 'presentation' for dim in story_data.dims]
        story_data = DataAssembly(story_data.values, coords=coords, dims=dims)
        story_data['story'] = 'presentation', [story] * len(story_data['presentation'])
        annotated_data.append(story_data)
    annotated_data = merge_data_arrays(annotated_data)
    return annotated_data


compare_characters = [',', '"', '\'', ':', '.', '!', '?', '(', ')']


def compare_ignore(sentence):
    for compare_character in compare_characters:
        sentence = sentence.replace(compare_character, '')
    sentence = sentence.replace('-', ' ')
    sentence = sentence.lower()
    return sentence


def _align_stimuli_recordings(stimulus_set, assembly):
    aligned_stimulus_set = []
    partial_sentences = assembly['stimulus_sentence'].values
    partial_sentences = [compare_ignore(sentence) for sentence in partial_sentences]
    assembly_stimset = {}
    stimulus_set_index = 0

    stories = ordered_set(assembly['story'].values.tolist())
    for story in tqdm(sorted(stories), desc='align stimuli', total=len(stories)):
        story_partial_sentences = [(sentence, i) for i, (sentence, sentence_story) in enumerate(zip(
            partial_sentences, assembly['story'].values)) if sentence_story == story]

        story_stimuli = stimulus_set[stimulus_set['story'] == story]
        stimuli_story = ' '.join(story_stimuli['sentence'])
        stimuli_story_sentence_starts = [0] + [len(sentence) + 1 for sentence in story_stimuli['sentence']]
        stimuli_story_sentence_starts = np.cumsum(stimuli_story_sentence_starts)
        assert ' '.join(s for s, i in story_partial_sentences) == compare_ignore(stimuli_story)
        stimulus_index = 0
        Stimulus = namedtuple('Stimulus', ['story', 'sentence', 'sentence_num', 'sentence_part'])
        sentence_parts = defaultdict(lambda: 0)
        for partial_sentence, assembly_index in story_partial_sentences:
            full_partial_sentence = ''
            partial_sentence_index = 0
            while partial_sentence_index < len(partial_sentence) \
                    or stimulus_index < len(stimuli_story) \
                    and stimuli_story[stimulus_index] in compare_characters + [' ']:
                if partial_sentence_index < len(partial_sentence) \
                        and partial_sentence[partial_sentence_index].lower() \
                        == stimuli_story[stimulus_index].lower():
                    full_partial_sentence += stimuli_story[stimulus_index]
                    stimulus_index += 1
                    partial_sentence_index += 1
                elif stimuli_story[stimulus_index] in compare_characters + [' ']:
                    # this case leads to a potential issue: Beginning quotations ' are always appended to
                    # the current instead of the next sentence. For now, I'm hoping this won't lead to issues.
                    full_partial_sentence += stimuli_story[stimulus_index]
                    stimulus_index += 1
                elif stimuli_story[stimulus_index] == '-':
                    full_partial_sentence += '-'
                    stimulus_index += 1
                    if partial_sentence[partial_sentence_index] == ' ':
                        partial_sentence_index += 1
                else:
                    raise NotImplementedError()
            sentence_num = next(index for index, start in enumerate(stimuli_story_sentence_starts)
                                if start >= stimulus_index) - 1
            sentence_part = sentence_parts[sentence_num]
            sentence_parts[sentence_num] += 1
            row = Stimulus(sentence=full_partial_sentence, story=story,
                           sentence_num=sentence_num, sentence_part=sentence_part)
            aligned_stimulus_set.append(row)
            assembly_stimset[assembly_index] = stimulus_set_index
            stimulus_set_index += 1
        # check
        aligned_story = "".join(row.sentence for row in aligned_stimulus_set if row.story == story)
        assert aligned_story == stimuli_story
    # build StimulusSet
    aligned_stimulus_set = StimulusSet(aligned_stimulus_set)
    aligned_stimulus_set['stimulus_id'] = [".".join([str(value) for value in values]) for values in zip(*[
        aligned_stimulus_set[coord].values for coord in ['story', 'sentence_num', 'sentence_part']])]
    aligned_stimulus_set.name = f"{stimulus_set.name}-aligned"

    # align assembly
    alignment = [stimset_idx for assembly_idx, stimset_idx in
                 sorted(assembly_stimset.items(), key=operator.itemgetter(0))]
    assembly_coords = {**{coord: (dims, values) for coord, dims, values in walk_coords(assembly)},
                       **{'stimulus_id': ('presentation', aligned_stimulus_set['stimulus_id'].values[alignment]),
                          'meta_sentence': ('presentation', assembly['stimulus_sentence'].values),
                          'sentence_number_in_story': ('presentation',
                                                       aligned_stimulus_set['sentence_num'].values[alignment]),
                          'part_number_in_sentence': ('presentation',
                                                      aligned_stimulus_set['sentence_part'].values[alignment]),
                          'stimulus_sentence': ('presentation', aligned_stimulus_set['sentence'].values[alignment]),
                          }}
    assembly = type(assembly)(assembly.values, coords=assembly_coords, dims=assembly.dims)
    assembly['stimulus'] = assembly['stimulus_sentence']

    return aligned_stimulus_set, assembly


def ordered_set(l):
    if isinstance(l, np.ndarray):
        l = l.tolist()
    return sorted(set(l), key=l.index)


def is_sorted(x):
    return all(x[i] <= x[i + 1] for i in range(len(x) - 1))


class NaturalisticStories:
    story_item_mapping = {'Boar': 1, 'Aqua': 2, 'MatchstickSeller': 3, 'KingOfBirds': 4, 'Elvis': 5,
                          'MrSticky': 6, 'HighSchool': 7, 'Roswell': 8, 'Tulips': 9, 'Tourette': 10}
    item_story_mapping = {item: story for story, item in story_item_mapping.items()}
    sentence_end = ['.', '?', '!', ".'", "?'", "!'"]

    def __init__(self, stimuli_filepath=Path(__file__).parent / 'naturalistic_stories' / 'all_stories.tok'):
        self._filepath = stimuli_filepath

    def __call__(self, keep_meta=True):
        data = pd.read_csv(self._filepath, delimiter='\t')

        def words_to_sentences(words):
            sentences = []
            sentence = ''
            for word in words:
                sentence += word
                if any(word.endswith(sentence_end) for sentence_end in self.sentence_end):
                    sentences.append(sentence)
                    sentence = ''
                else:
                    sentence += ' '
            return pd.DataFrame({'sentence': sentences, 'sentence_num': list(range(len(sentences)))})

        data = data.groupby('item')['word'].apply(words_to_sentences).reset_index(level=0)
        data['story'] = [self.item_story_mapping[item] for item in data['item']]
        data = data[['story', 'item', 'sentence_num', 'sentence']]
        data = StimulusSet(data)
        data.name = 'naturalistic_stories'
        return data


def average_subregions(assembly):
    attrs = assembly.attrs
    del assembly['threshold']
    # group by stimuli, fROI, subject after one another.
    # this gets rid of adjacent coords unfortunately, but we accept that for now.
    averaged_assembly = assembly.groupby('stimulus_id').apply(
        lambda stimulus_group: stimulus_group.groupby('fROI_area').apply(
            lambda fROI_group: fROI_group.groupby('subject_id').mean()
        ))
    averaged_assembly = averaged_assembly.stack(neuroid=['fROI_area', 'subject_id'])
    # copy presentation coords back since those are needed for e.g. metric stratification
    order = [averaged_assembly['stimulus_id'].values.tolist().index(stimulus_id)
             for stimulus_id in assembly['stimulus_id'].values]
    for copy_coord, dims, copy_value in walk_coords(assembly):
        if not array_is_element(dims, 'presentation') or hasattr(averaged_assembly, copy_coord):
            continue
        averaged_assembly[copy_coord] = dims, copy_value[order]
    averaged_assembly.attrs = attrs
    averaged_assembly['neuroid_id'] = 'neuroid', [".".join([str(value) for value in values]) for values in zip(*[
        averaged_assembly[coord].values for coord in ['subject_id', 'fROI_area']])]
    return averaged_assembly


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    upload_blank2014()
