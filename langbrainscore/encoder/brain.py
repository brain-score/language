from langbrainscore.interface.encoder import _BrainEncoder

import langbrainscore

class BrainEncoder(_BrainEncoder):
    '''
    This class provides a wrapper around a brain Dataset object
    that merely checks a few assertions and returns its contents,
    but is used to maintain the Encoder interface.
    '''

    def __init__(self) -> None:
        pass


    def encode(self, dataset: 'langbrainscore.dataset.Dataset', average_time = False):
        """returns an "encoding" of stimuli (passed in as a Dataset)

        Args:
            langbrainscore.dataset.Dataset: brain dataset object

        Returns:
            xr.DataArray: contents of brain dataset
        """
        if not isinstance(dataset, langbrainscore.dataset.Dataset):
            raise TypeError(f"dataset must be of type `langbrainscore.dataset.Dataset`, not {type(dataset)}")
        if average_time:
            return (
                dataset._dataset
                .mean('timeid')
                .expand_dims('timeid', 2)
                .assign_coords({'timeid': ('timeid', [0])})
            )
        return dataset._dataset
