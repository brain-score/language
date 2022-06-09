import os
import typing

import numpy as np
import xarray as xr
from brainscore_language.dataset import Dataset
from brainscore_language.interface import EncoderRepresentations, _ModelEncoder
from brainscore_language.utils.encoder import (
    flatten_activations_per_sample,
    get_context_groups,
    get_torch_device,
    postprocess_activations,
    repackage_flattened_activations,
    encode_stimuli_in_context,
)
from brainscore_language.utils.logging import log
from brainscore_language.utils.resources import config_name_mappings
from tqdm import tqdm

from brainscore_language.utils.xarray import copy_metadata, fix_xr_dtypes

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class HuggingFaceEncoder(_ModelEncoder):
    def __init__(
            self,
            model_id: str,
            emb_aggregation: typing.Union[str, None, typing.Callable],
            device=get_torch_device(),
            context_dimension: str = None,
            bidirectional: bool = False,
            emb_preproc: typing.Tuple[str] = (),
            include_special_tokens: bool = True,
    ):
        """
        Args:
            model_id: the model id
            device: the device to use
            context_dimension: the dimension to use for extracting strings using context.
                if None, each sampleid (stimuli) will be treated as a single context group.
                if a string is specified, the string must refer to the name of a dimension in the xarray-like dataset
                object (langbrainscore.dataset.Dataset) that provides groupings of sampleids (stimuli) that should be
                used as context when generating encoder representations.
            bidirectional: whether to use bidirectional encoder (i.e., access both forward and backward context)
            emb_aggregation: how to aggregate the hidden states of
                the encoder representations for each sampleid (stimuli).
            emb_preproc: a list of strings specifying preprocessing functions to apply to the aggregated embeddings.
                Processing is performed layer-wise.
            include_special_tokens: whether to include special tokens in the encoder representations.
        """

        super().__init__(
            model_id,
            _context_dimension=context_dimension,
            _bidirectional=bidirectional,
            _emb_aggregation=emb_aggregation,
            _emb_preproc=emb_preproc,
            _include_special_tokens=include_special_tokens,
        )

        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers import logging as transformers_logging

        transformers_logging.set_verbosity_error()

        self.device = device or get_torch_device()
        self.config = AutoConfig.from_pretrained(self._model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, multiprocessing=True
        )
        self.model_specs = None
        self.model = AutoModel.from_pretrained(self._model_id, config=self.config)
        try:
            self.model = self.model.to(self.device)
        except RuntimeError:
            self.device = "cpu"
            self.model = self.model.to(self.device)

    def get_encoder_representations_template(
            self, dataset=None, representations=xr.DataArray()
    ) -> EncoderRepresentations:
        """
        returns an empty `EncoderRepresentations` object with all the appropriate
        attributes but the `dataset` and `representations` missing and to be filled in
        later.
        """
        return EncoderRepresentations(
            dataset=dataset,
            representations=representations,
            model_id=self._model_id,
            context_dimension=self._context_dimension,
            bidirectional=self._bidirectional,
            emb_aggregation=self._emb_aggregation,
            emb_preproc=self._emb_preproc,
            include_special_tokens=self._include_special_tokens,
        )

    def encode(
            self,
            dataset: Dataset,
            read_cache: bool = True,  # avoid recomputing if cached `EncoderRepresentations` exists, recompute if not
            write_cache: bool = True,  # dump the result of this computation to cache?
    ) -> EncoderRepresentations:
        """
        Input a langbrainscore Dataset, encode the stimuli according to the parameters specified in init, and return
            the an xarray DataArray of aggregated representations for each stimulus.

        Args:
            dataset (langbrainscore.dataset.DataSet): [description]
            read_cache (bool): Avoid recomputing if cached `EncoderRepresentations` exists, recompute if not
            write_cache (bool): Dump and write the result of the computed encoder representations to cache

        Raises:
            NotImplementedError: [description]
            ValueError: [description]

        Returns:
            [type]: [description]
        """

        # before computing the representations from scratch, we will first see if any
        # cached representations exist already.

        if read_cache:
            to_check_in_cache: EncoderRepresentations = (
                self.get_encoder_representations_template(dataset=dataset)
            )

            try:
                to_check_in_cache.load_cache()
                return to_check_in_cache
            except FileNotFoundError:
                log(
                    f"couldn't load cached reprs for {to_check_in_cache.identifier_string}; recomputing.",
                    cmap="WARN",
                    type="WARN",
                )

        self.model.eval()
        stimuli = dataset.stimuli.values

        # Initialize the context group coordinate (obtain embeddings with context)
        context_groups = get_context_groups(dataset, self._context_dimension)

        # list for storing activations for each stimulus with all layers flattened
        # list for storing layer ids ([0 0 0 0 ... 1 1 1 ...]) indicating which layer each
        # neuroid (representation dimension) came from
        flattened_activations, layer_ids = [], []

        ###############################################################################
        # ALL SAMPLES LOOP
        ###############################################################################
        _, unique_ixs = np.unique(context_groups, return_index=True)
        # Make sure context group order is preserved
        for group in tqdm(context_groups[np.sort(unique_ixs)], desc="Encoding stimuli"):
            # Mask based on the context group
            mask_context = context_groups == group
            stimuli_in_context = stimuli[mask_context]

            # store model states for each stimulus in this context group
            encoded_stimuli = []

            ###############################################################################
            # CONTEXT LOOP
            ###############################################################################
            for encoded_stim in encode_stimuli_in_context(
                    stimuli_in_context=stimuli_in_context,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    bidirectional=self._bidirectional,
                    include_special_tokens=self._include_special_tokens,
                    emb_aggregation=self._emb_aggregation,
                    device=self.device,
            ):
                encoded_stimuli += [encoded_stim]
            ###############################################################################
            # END CONTEXT LOOP
            ###############################################################################

            # Flatten activations across layers and package as xarray
            flattened_activations_and_layer_ids = [
                *map(flatten_activations_per_sample, encoded_stimuli)
            ]
            for f_as, l_ids in flattened_activations_and_layer_ids:
                flattened_activations += [f_as]
                layer_ids += [l_ids]
                assert len(f_as) == len(l_ids)  # Assert all layer lists are equal

        ###############################################################################
        # END ALL SAMPLES LOOP
        ###############################################################################

        # Stack flattened activations and layer ids to obtain [n_samples, emb_din * n_layers]
        activations_2d = np.vstack(flattened_activations)
        layer_ids_1d = np.squeeze(np.unique(np.vstack(layer_ids), axis=0))

        # Post-process activations after obtaining them (or "pre-process" them before computing brainscore)
        if len(self._emb_preproc) > 0:
            for mode in self._emb_preproc:
                activations_2d, layer_ids_1d = postprocess_activations(
                    activations_2d=activations_2d,
                    layer_ids_1d=layer_ids_1d,
                    emb_preproc_mode=mode,
                )

        assert activations_2d.shape[1] == len(layer_ids_1d)
        assert activations_2d.shape[0] == len(stimuli)

        # Package activations as xarray and reapply metadata
        encoded_dataset: xr.DataArray = repackage_flattened_activations(
            activations_2d=activations_2d,
            layer_ids_1d=layer_ids_1d,
            dataset=dataset,
        )
        encoded_dataset: xr.DataArray = copy_metadata(
            encoded_dataset,
            dataset.contents,
            "sampleid",
        )

        to_return = EncoderRepresentations(
            dataset=dataset,
            representations=encoded_dataset,
            context_dimension=self._context_dimension,
            bidirectional=self._bidirectional,
            emb_aggregation=self._emb_aggregation,
            emb_preproc=self._emb_preproc,
            include_special_tokens=self._include_special_tokens,
        )

        if write_cache:
            to_return.to_cache(overwrite=True)

        return to_return

    def get_modelcard(self):
        """
        Returns the model card of the model (model-wise, and not layer-wise)
        """

        model_classes = [
            "gpt",
            "bert",
        ]  # continuously update based on new model classes supported

        # based on the model_id, figure out which model class it is
        model_class = [x for x in model_classes if x in self._model_id][0]
        assert model_class is not None, f"model_id {self._model_id} not supported"

        config_specs_of_interest = config_name_mappings[model_class]

        model_specs = {}
        for (
                k_spec,
                v_spec,
        ) in (
                config_specs_of_interest.items()
        ):  # key is the name we want to use in the model card,
            # value is the name in the config
            if v_spec is not None:
                model_specs[k_spec] = getattr(self.config, v_spec)
            else:
                model_specs[k_spec] = None

        self.model_specs = model_specs

        return model_specs
