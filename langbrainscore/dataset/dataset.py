import typing
import numpy as np
import pandas as pd

class Dataset:
    _stimuli_path = None
    _stimuli = None
    _stim_metadata = None

    def __init__(self, stimuli: typing.Union[list, np.array, pd.DataFrame], 
                #  stimuli_path: str = None,
                 stim_metadata: typing.Union[pd.DataFrame] = None) -> None:

        self._stimuli = stimuli

        if stim_metadata:
            self._stim_metadata = stim_metadata

    @property
    def stimuli(self):
        return self._stimuli


class BrainDataset(Dataset):
    '''
    A subclass of `Dataset` to support storage and retrieval of brain data.
    "Brain data" broadly refers to _real-world_ data collected from individuals with a brain,
    and may be of the following forms:
            - neuroimaging [fMRI, PET]
            - physiological [ERP, MEG, ECOG]
            - behavioral [RT, Eye-tracking]
    In general, data must be supplied in the mapping
            stimulus --> neuroid
        where `neuroid` is the basic unit of brain recording (it may be a voxel, an electrode, etc).

    Inputs:
        recorded_data: <n_stimuli> x <n_neuroids>
        metadata: <n_stimuli> x -1
    '''
    _recorded_data = None
    _recording_metadata = None

    def __init__(self,
                 stimuli: typing.Union[list, pd.DataFrame, np.array],
                 recorded_data: typing.Union[np.ndarray, pd.DataFrame, str],
                 stimuli_metadata: typing.Union[pd.DataFrame, str] = None,
                 recording_metadata: typing.Union[pd.DataFrame, str] = None) -> None:
                 
        # TODOs for the (distant) future:
        # scipy.io.loadmat stuf
        # h5py 2 GB limit issue
        
        super().__init__(stimuli, stimuli_metadata)
        self._recorded_data = recorded_data
    
        if not recording_metadata is None: 
          # ^-- may not be necessary; users should always provide subject_ids
          # for cross-validation 
            self._recording_metadata = recording_metadata
        else:
            raise ValueError(f'forgot to pass recording metadata?')

        # make sure the data dimensions make sense
        self.num_stimuli = len(self.stimuli)
        self.num_neuroids = self._recorded_data.shape[1]

        # recorded_data contains a row per stimulus
        assert self.num_stimuli == self._recorded_data.shape[0]
        # recording_metadata contains a row per neuroid
        assert self.num_neuroids == self._recording_metadata.shape[0]


    @property
    def recorded_data(self) -> np.array:
        if type(self._recorded_data) not in {np.array, np.ndarray, }:
            # TODO: standardize output format for the recorded data 
            raise NotImplementedError(f'recorded_data of type {type(self._recorded_data)} cannot be returned')
        return self._recorded_data

    @property
    def recording_metadata(self):
        return self._recording_metadata
