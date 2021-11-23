import typing
import numpy as np
import pandas as pd
import xarray as xr

class Dataset:
    _stimuli_path = None
    _stimuli = None
    _stim_metadata = None

    def __init__(self, xr_dataset: xr.Dataset) -> None:
        # TODO: the below is no longer true
        """initializer method that accepts an xarray with at least the following 
            core coordinates: sampleid, neuroid, timeid

            https://docs.google.com/document/d/10U9cPphFQdXcCDQtq-yTqbB_jLa-k4hy5pe9VoD08Sg/edit?usp=sharing

        Args:
            xr_dataset (xr.Dataset): Xarray dataset object with minimal core dimensions
        """        
        # set the internal `_xr_dataset` reference to the one passed in to this method 
        self._xr_dataset = xr_dataset


    @property
    def _dataset(self):
        '''
        returns the internal xarray dataset object. use with caution.
        '''
        return self._xr_dataset


    @property
    def stimuli(self) -> typing.Union[np.ndarray, xr.DataArray]:
        """getter method that returns an ndarray-like object of stimuli

        Returns:
            typing.Union[np.ndarray, xr.DataArray]: array-like object containing the stimuli from the dataset
        """        
        return self._xr_dataset.stimuli

    @property
    def dims(self) -> tuple:
        '''
        '''
        return self._xr_dataset.dims


# TODO: we should implement the packaging into a xarray dataset all in here ^ and below, not expect a pre-packaged xarray.
# we will minimally require only stimuli and (below:) recorded data, and then construct an xarray internally
# NOTE: this is to protect users from the low-level details of our data packaging.

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
                #  stimuli: typing.Union[list, pd.DataFrame, np.array],
                #  recorded_data: typing.Union[np.ndarray, pd.DataFrame, str],
                #  stimuli_metadata: typing.Union[pd.DataFrame, str] = None,
                #  recording_metadata: typing.Union[pd.DataFrame, str] = None
                ) -> None:
        pass

        # TODOs for the (distant) future:
        # scipy.io.loadmat stuf
        # h5py 2 GB limit issue
        
        # super().__init__()
        # self._recorded_data = recorded_data
    
        # if recording_metadata is not None: 
        #   # ^-- may not be necessary; users should always provide subject_ids
        #   # for cross-validation 
        #     self._recording_metadata = recording_metadata
        # else:
        #     raise ValueError(f'forgot to pass recording metadata?')

        # # make sure the data dimensions make sense
        # self.num_stimuli = len(self.stimuli)
        # self.num_neuroids = self._recorded_data.shape[1]

        # # recorded_data contains a row per stimulus
        # assert self.num_stimuli == self._recorded_data.shape[0]
        # # recording_metadata contains a row per neuroid
        # assert self.num_neuroids == self._recording_metadata.shape[0]


    @property
    def recorded_data(self) -> np.array:
        if type(self._recorded_data) not in {np.array, np.ndarray, }:
            # TODO: standardize output format for the recorded data 
            raise NotImplementedError(f'recorded_data of type {type(self._recorded_data)} cannot be returned')
        return self._recorded_data

    @property
    def recording_metadata(self):
        return self._recording_metadata
