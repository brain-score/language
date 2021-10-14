import typing

class Dataset:
    _stimuli_path_or_uri = None
    _stimuli = None

    def __init__(self) -> None:
        ...

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
    '''
    _data_path_or_uri = None
    _recorded_data = None

    def __init__(self, data_path_or_uri) -> None:
        _data_path_or_uri = data_path_or_uri

    @property
    def recorded_data(self):
        ...

