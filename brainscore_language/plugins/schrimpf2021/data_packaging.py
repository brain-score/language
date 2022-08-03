from glob import glob

import logging
import numpy as np
import os
import re
import scipy.io
import scipy.stats
import sys
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from brainio import fetch
from brainio.assemblies import NeuroidAssembly, walk_coords, merge_data_arrays
from brainio.packaging import write_netcdf, upload_to_s3

_logger = logging.getLogger(__name__)

"""
The code in this file was run only once to initially upload the data, and is kept here for reference.
"""


# adapted from
# https://github.com/mschrimpf/neural-nlp/blob/cedac1f868c8081ce6754ef0c13895ce8bc32efc/neural_nlp/neural_data/fmri.py

def load_Pereira2018():
    data_dir = Path(__file__).parent / "Pereira2018"
    experiment2, experiment3 = "243sentences.mat", "384sentences.mat"
    stimuli = {}  # experiment -> stimuli
    assemblies = []
    subject_directories = [d for d in data_dir.iterdir() if d.is_dir()]
    for subject_directory in tqdm(subject_directories, desc="subjects"):
        for experiment_filename in [experiment2, experiment3]:
            data_file = subject_directory / f"examples_{experiment_filename}"
            if not data_file.is_file():
                _logger.debug(f"{subject_directory} does not contain {experiment_filename}")
                continue
            data = scipy.io.loadmat(str(data_file))

            # assembly
            assembly = data['examples']
            meta = data['meta']
            assembly = NeuroidAssembly(assembly, coords={
                'experiment': ('presentation', [os.path.splitext(experiment_filename)[0]] * assembly.shape[0]),
                'stimulus_num': ('presentation', list(range(assembly.shape[0]))),
                'passage_index': ('presentation', data['labelsPassageForEachSentence'][:, 0]),
                'passage_label': ('presentation', [data['keyPassages'][index - 1, 0][0]
                                                   for index in data['labelsPassageForEachSentence'][:, 0]]),
                'passage_category': ('presentation', [
                    data['keyPassageCategory'][0, data['labelsPassageCategory'][index - 1, 0] - 1][0][0]
                    for index in data['labelsPassageForEachSentence']]),

                'subject': ('neuroid', [subject_directory.name] * assembly.shape[1]),
                'voxel_num': ('neuroid', list(range(assembly.shape[1]))),
                'AAL_roi_index': ('neuroid', meta[0][0]['roiMultimaskAAL'][:, 0]),
            }, dims=['presentation', 'neuroid'])
            stimulus_id = _build_id(assembly, ['experiment', 'stimulus_num'])
            assembly['stimulus_id'] = 'presentation', stimulus_id
            # set story for compatibility
            assembly['story'] = 'presentation', _build_id(assembly, ['experiment', 'passage_category'])
            assembly['neuroid_id'] = 'neuroid', _build_id(assembly, ['subject', 'voxel_num'])
            assemblies.append(assembly)

    _logger.debug(f"Merging {len(assemblies)} assemblies")
    assembly = merge_data_arrays(assemblies)

    _logger.debug("Creating StimulusSet")
    return assembly


def upload_pereira2018():
    # adapted from
    # https://github.com/mschrimpf/neural-nlp/blob/cedac1f868c8081ce6754ef0c13895ce8bc32efc/neural_nlp/neural_data/naturalStories.py#L15

    reference_data = load_Pereira2018()

    data_dir = Path(__file__).parent / "Pereira2018_Blank"
    experiments = {'n72': "243sentences", 'n96': "384sentences"}
    assemblies = []
    subjects = ['018', '199', '215', '288', '289', '296', '343', '366', '407', '426']
    for subject in tqdm(subjects, desc="subjects"):
        subject_assemblies = []
        for experiment_filepart, experiment_name in experiments.items():
            filepath = data_dir / f"{subject}_complang_passages_{experiment_filepart}_persent.mat"
            if not filepath.is_file():
                _logger.debug(f"Subject {subject} did not run {experiment_name}: {filepath} does not exist")
                continue
            data = scipy.io.loadmat(str(filepath))

            # construct assembly
            assembly = data['data']
            neuroid_meta = data['meta']

            expanded_assembly = []
            voxel_nums, atlases, filter_strategies, atlas_selections, atlas_filter_lower, rois = [], [], [], [], [], []
            for voxel_num in range(assembly.shape[1]):
                for atlas_iter, atlas_selection in enumerate(neuroid_meta['atlases'][0, 0][:, 0]):
                    multimask = neuroid_meta['roiMultimask'][0, 0][atlas_iter, 0][voxel_num, 0]
                    if np.isnan(multimask):
                        continue
                    atlas_selection = atlas_selection[0].split('_')
                    filter_strategy = None if len(atlas_selection) != 3 else atlas_selection[1]
                    filter_lower = re.match(r'from([0-9]{2})to100prcnt', atlas_selection[-1])
                    atlas_filter_lower.append(int(filter_lower.group(1)))
                    atlas, selection = atlas_selection[0], atlas_selection[-1]
                    atlases.append(atlas)
                    filter_strategies.append(filter_strategy)
                    atlas_selections.append(selection)
                    multimask = int(multimask) - 1  # Matlab 1-based to Python 0-based indexing
                    rois.append(neuroid_meta['rois'][0, 0][atlas_iter, 0][multimask, 0][0])
                    voxel_nums.append(voxel_num)
                    expanded_assembly.append(assembly[:, voxel_num])
            # ensure all are mapped
            assert set(voxel_nums) == set(range(assembly.shape[1])), "not all voxels mapped"
            # add indices
            assembly = np.stack(expanded_assembly).T
            assert assembly.shape[1] == len(atlases) == len(atlas_selections) == len(rois)
            indices_in_3d = neuroid_meta['indicesIn3D'][0, 0][:, 0]
            indices_in_3d = [indices_in_3d[voxel_num] for voxel_num in voxel_nums]
            # add coords
            col_to_coords = np.array([neuroid_meta['colToCoord'][0, 0][voxel_num] for voxel_num in voxel_nums])

            # put it all together
            assembly = NeuroidAssembly(assembly, coords={
                **{coord: (dims, value) for coord, dims, value in walk_coords(
                    reference_data.sel(experiment=experiment_name)['presentation'])},
                **{'experiment': ('presentation', [experiment_name] * assembly.shape[0]),
                   'subject': ('neuroid', [subject] * assembly.shape[1]),
                   'voxel_num': ('neuroid', voxel_nums),
                   'atlas': ('neuroid', atlases),
                   'filter_strategy': ('neuroid', filter_strategies),
                   'atlas_selection': ('neuroid', atlas_selections),
                   'atlas_selection_lower': ('neuroid', atlas_filter_lower),
                   'roi': ('neuroid', rois),
                   'indices_in_3d': ('neuroid', indices_in_3d),
                   'col_to_coord_1': ('neuroid', col_to_coords[:, 0]),
                   'col_to_coord_2': ('neuroid', col_to_coords[:, 1]),
                   'col_to_coord_3': ('neuroid', col_to_coords[:, 2]),
                   }}, dims=['presentation', 'neuroid'])
            assembly['neuroid_id'] = 'neuroid', _build_id(assembly, ['subject', 'voxel_num'])
            subject_assemblies.append(assembly)
        assembly = merge_data_arrays(subject_assemblies)
        assemblies.append(assembly)

    _logger.debug(f"Merging {len(assemblies)} assemblies")
    assembly = merge_data_arrays(assemblies)

    # filter
    assembly = assembly.sel(atlas='language', atlas_selection_lower=90)
    assembly = assembly[{'neuroid': [filter_strategy in [np.nan, 'HminusE', 'FIXminusH']
                                     for filter_strategy in assembly['filter_strategy'].values]}]

    # upload
    upload_data_assembly(assembly,
                         assembly_identifier="Pereira2018.language_system",
                         bucket_name="brainscore-language")


# adapted from
# https://github.com/mschrimpf/neural-nlp/blob/cedac1f868c8081ce6754ef0c13895ce8bc32efc/neural_nlp/neural_data/ecog.py#L18

def load_Fedorenko2016(electrodes='language', version=3):
    ressources_dir = Path(__file__).parent.parent.parent / 'ressources'
    neural_data_dir = ressources_dir / 'neural_data' / 'ecog-Fedorenko2016/'
    stim_data_dir = ressources_dir / 'stimuli' / 'sentences_8'
    _logger.info(f'Neural data directory: {neural_data_dir}')
    filepaths_stim = glob(os.path.join(stim_data_dir, '*.txt'))

    # ECoG
    data = None

    # For language responsive electrodes:
    if electrodes == 'language':

        # Create a subject ID list corresponding to language electrodes
        subject1 = np.repeat(1, 47)
        subject2 = np.repeat(2, 9)
        subject3 = np.repeat(3, 9)
        subject4 = np.repeat(4, 15)
        subject5 = np.repeat(5, 18)

        if version == 1:
            filepath_neural = glob(os.path.join(neural_data_dir, '*ecog.mat'))

        if version == 2:
            filepath_neural = glob(os.path.join(neural_data_dir, '*metadata_lang.mat'))

        if version == 3:
            subject1 = np.repeat(1, 47)
            subject2 = np.repeat(2, 8)
            subject3 = np.repeat(3, 9)
            subject4 = np.repeat(4, 15)
            subject5 = np.repeat(5, 18)

            filepath_neural = glob(os.path.join(neural_data_dir, '*g_lang_v3.mat'))

        if version == 4:
            subject1 = np.repeat(1, 49)
            subject2 = np.repeat(2, 8)
            subject3 = np.repeat(3, 10)
            subject4 = np.repeat(4, 16)
            subject5 = np.repeat(5, 19)
            subject6 = np.repeat(6, 3)

            filepath_neural = glob(os.path.join(neural_data_dir, '*g_lang_v4.mat'))

        _logger.debug(f'Running Fedorenko2016 benchmark with language responsive electrodes, data version: {version}')

    # For non-noisy electrodes
    if electrodes == 'all':

        # Create a subject ID list corresponding to all electrodes
        subject1 = np.repeat(1, 70)
        subject2 = np.repeat(2, 35)
        subject3 = np.repeat(3, 20)
        subject4 = np.repeat(4, 29)
        subject5 = np.repeat(5, 26)

        if version == 1:
            filepath_neural = glob(os.path.join(neural_data_dir, '*ecog_all.mat'))

        if version == 2:
            filepath_neural = glob(os.path.join(neural_data_dir, '*metadata_all.mat'))

        if version == 3:
            subject1 = np.repeat(1, 67)
            subject2 = np.repeat(2, 35)
            subject3 = np.repeat(3, 20)
            subject4 = np.repeat(4, 29)
            subject5 = np.repeat(5, 26)

            filepath_neural = glob(os.path.join(neural_data_dir, '*all_v3.mat'))

        if version == 4:
            subject1 = np.repeat(1, 63)
            subject2 = np.repeat(2, 35)
            subject3 = np.repeat(3, 21)
            subject4 = np.repeat(4, 29)
            subject5 = np.repeat(5, 27)
            subject6 = np.repeat(6, 9)

            filepath_neural = glob(os.path.join(neural_data_dir, '*all_v4.mat'))

        _logger.debug('Running Fedorenko2016 benchmark with non-noisy electrodes, data version: ', version)

        # For non-noisy electrodes
    if electrodes == 'non-language':

        if version == 1 or version == 2:
            filepath_neural = glob(os.path.join(neural_data_dir, '*nonlang.mat'))

            # Create a subject ID list corresponding to non-language electrodes
            subject1 = np.repeat(1, 28)
            subject2 = np.repeat(2, 31)
            subject3 = np.repeat(3, 14)
            subject4 = np.repeat(4, 19)
            subject5 = np.repeat(5, 16)

        if version == 3:
            filepath_neural = glob(os.path.join(neural_data_dir, '*nonlang_v3.mat'))

            # Create a subject ID list corresponding to non-language electrodes
            subject1 = np.repeat(1, 25)  # 47 lang selective,
            subject2 = np.repeat(2, 31)
            subject3 = np.repeat(3, 14)
            subject4 = np.repeat(4, 19)
            subject5 = np.repeat(5, 16)  # 10 lang electrodes in the non-noisy

        if version == 4:
            filepath_neural = glob(os.path.join(neural_data_dir, '*nonlang_v4.mat'))

            # Create a subject ID list corresponding to non-language electrodes
            subject1 = np.repeat(1, 22)
            subject2 = np.repeat(2, 31)
            subject3 = np.repeat(3, 15)
            subject4 = np.repeat(4, 19)
            subject5 = np.repeat(5, 18)
            subject6 = np.repeat(6, 6)

        _logger.debug(f'Running Fedorenko2016 benchmark with non-language electrodes, data version: {version}')

    ecog_mat = scipy.io.loadmat(filepath_neural[0])
    ecog_mtrix = ecog_mat['ecog']

    if version == 1:  # Manually z-score the version 1 data
        ecog_z = scipy.stats.zscore(ecog_mtrix, 1)
    if version == 2 or version == 3 or version == 4:
        ecog_z = ecog_mtrix

    ecog_mtrix_T = np.transpose(ecog_z)

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

    if version == 1 or version == 2 or version == 3:
        subjectID = np.concatenate([subject1, subject2, subject3, subject4, subject5], axis=0)

    if version == 4:
        subjectID = np.concatenate([subject1, subject2, subject3, subject4, subject5, subject6], axis=0)

    # Create a list for each word number
    word_number = list(range(np.shape(ecog_mtrix_T)[0]))

    # Add a pd df as the stimulus_set
    zipped_lst = list(zip(sentenceID, word_number, sentence_words))
    df_stimulus_set = StimulusSet(zipped_lst, columns=['sentence_id', 'stimulus_id', 'word'])
    df_stimulus_set.name = 'Fedorenko2016.ecog'

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

    assembly.attrs['stimulus_set'] = df_stimulus_set  # Add the stimulus_set dataframe
    data = assembly if data is None else xr.concat(data, assembly)
    return NeuroidAssembly(data)


def _build_id(assembly, coords):
    return [".".join([f"{value}" for value in values]) for values in zip(*[assembly[coord].values for coord in coords])]


def upload_data_assembly(assembly, assembly_identifier, bucket_name):
    # adapted from
    # https://github.com/mschrimpf/brainio/blob/8a40a3558d0b86072b9e221808f19005c7cb8c17/brainio/packaging.py#L217

    _logger.debug(f"Uploading {assembly_identifier} to S3")

    # identifiers
    assembly_store_identifier = "assy_" + assembly_identifier.replace(".", "_")
    netcdf_file_name = assembly_store_identifier + ".nc"
    target_netcdf_path = Path(fetch.get_local_data_path()) / assembly_store_identifier / netcdf_file_name
    s3_key = netcdf_file_name

    # write to disk and upload
    netcdf_kf_sha1 = write_netcdf(assembly, target_netcdf_path)
    upload_to_s3(target_netcdf_path, bucket_name, s3_key)
    _logger.debug(f"Uploaded assembly {assembly_identifier} to S3: {s3_key} (SHA1 hash {netcdf_kf_sha1})")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    upload_pereira2018()
