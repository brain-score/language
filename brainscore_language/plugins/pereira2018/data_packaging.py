import logging
import numpy as np
import os
import re
import scipy.io
import scipy.stats
import sys
from pathlib import Path
from tqdm import tqdm

from brainio.assemblies import NeuroidAssembly, walk_coords, merge_data_arrays
from brainscore_language.utils.s3 import upload_data_assembly

_logger = logging.getLogger(__name__)

"""
The code in this file was run only once to initially upload the data, and is kept here for reference.
"""


def upload_pereira2018():
    assembly = load_pereira2018()
    upload_data_assembly(assembly,
                         assembly_identifier="Pereira2018.language",
                         bucket_name="brainscore-language")


# adapted from
# https://github.com/mschrimpf/neural-nlp/blob/cedac1f868c8081ce6754ef0c13895ce8bc32efc/neural_nlp/neural_data/fmri.py

def load_Pereira2018_reference():
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
            sentences = [sentence.item() for sentence in data['keySentences'][:, 0]]
            assembly = NeuroidAssembly(assembly, coords={
                'experiment': ('presentation', [os.path.splitext(experiment_filename)[0]] * assembly.shape[0]),
                'stimulus_num': ('presentation', list(range(assembly.shape[0]))),
                'sentence': ('presentation', sentences),
                'stimulus': ('presentation', sentences),
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
    return assembly


def load_pereira2018():
    # adapted from
    # https://github.com/mschrimpf/neural-nlp/blob/cedac1f868c8081ce6754ef0c13895ce8bc32efc/neural_nlp/neural_data/naturalStories.py#L15

    reference_data = load_Pereira2018_reference()

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

    return assembly


def _build_id(assembly, coords):
    return [".".join([f"{value}" for value in values]) for values in zip(*[assembly[coord].values for coord in coords])]


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    upload_pereira2018()
