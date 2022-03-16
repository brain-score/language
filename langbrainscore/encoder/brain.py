import langbrainscore
import xarray as xr
from langbrainscore.interface.encoder import _BrainEncoder


class BrainEncoder(_BrainEncoder):
    """
    This class provides a wrapper around a brain Dataset object
    that merely checks a few assertions and returns its contents,
    but is used to maintain the Encoder interface.
    """

    def __init__(self) -> "BrainEncoder":
        pass

    def encode(
        self, dataset: langbrainscore.dataset.Dataset, average_time: bool = False,
    ) -> xr.DataArray:
        """returns an "encoding" of stimuli (passed in as a Dataset)

        Args:
            langbrainscore.dataset.Dataset: brain dataset object

        Returns:
            xr.DataArray: contents of brain dataset
        """
        if not isinstance(dataset, langbrainscore.dataset.Dataset):
            raise TypeError(
                f"dataset must be of type `langbrainscore.dataset.Dataset`, not {type(dataset)}"
            )
        if average_time:
            dim = "timeid"
            return (
                dataset._dataset.mean(dim)
                .expand_dims(dim, 2)
                .assign_coords({dim: (dim, [0])})
            )
        return dataset._dataset
