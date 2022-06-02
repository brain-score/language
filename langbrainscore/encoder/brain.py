import langbrainscore
import xarray as xr
from langbrainscore.interface.encoder import _Encoder, EncoderRepresentations


class BrainEncoder(_Encoder):
    """
    This class is used to extract the relevant contents of a given
    `langbrainscore.dataset.Dataset` object and maintains the Encoder interface.
    """

    def __init__(
        self, measurement: str = "unknown", aggregate_time: bool = False
    ) -> "BrainEncoder":
        """Initialize a BrainEncoder

        Args:
            modality (str, optional): The modality/type of human data. Defaults to None.
            aggregate_time (bool, optional): Whether we should aggregate timeid dimension of the
                data during encoding. Defaults to False.

        Returns:
            BrainEncoder: _description_
        """
        self._measurement = measurement
        self._aggregate_time = aggregate_time

    def encode(
        self,
        dataset: langbrainscore.dataset.Dataset,
    ) -> EncoderRepresentations:
        """
        returns human measurements related to stimuli (passed in as a Dataset)

        Args:
            langbrainscore.dataset.Dataset: brain dataset object

        Returns:
            xr.DataArray: contents of brain dataset
        """
        self._check_dataset_interface(dataset)
        if self._aggregate_time:
            dim = "timeid"
            return (
                dataset.contents.mean(dim)
                .expand_dims(dim, 2)
                .assign_coords({dim: (dim, [0])})
            )

        if "measurement" in dataset.contents.attrs:
            self._measurement = dataset.contents.attrs["measurement"]

        # return dataset.contents
        return EncoderRepresentations(
            dataset=dataset,
            representations=dataset.contents,
            model_id=self._measurement,
            emb_aggregation=None,
            emb_preproc=(),
            include_special_tokens=None,
            context_dimension=None,
            bidirectional=False,
        )
