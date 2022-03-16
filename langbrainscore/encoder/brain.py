import langbrainscore
import xarray as xr
from langbrainscore.interface.encoder import _Encoder


class BrainEncoder(_Encoder):
    """
    This class is used to extract the relevant contents of a given
    `langbrainscore.dataset.Dataset` object and maintains the Encoder interface.
    """

    def __init__(self) -> "BrainEncoder":
        pass

    def encode(
        self, dataset: langbrainscore.dataset.Dataset, average_time: bool = False,
    ) -> xr.DataArray:
        """
        returns human measurements related to stimuli (passed in as a Dataset)

        Args:
            langbrainscore.dataset.Dataset: brain dataset object

        Returns:
            xr.DataArray: contents of brain dataset
        """
        self._check_dataset_interface(dataset)
        if average_time:
            dim = "timeid"
            return (
                dataset.contents.mean(dim)
                .expand_dims(dim, 2)
                .assign_coords({dim: (dim, [0])})
            )
        return dataset.contents
