import typing
from enum import unique

import numpy as np
import torch
from tqdm import tqdm
import xarray as xr

from langbrainscore.dataset import Dataset
from langbrainscore.interface import EncoderRepresentations, _ModelEncoder
from langbrainscore.utils.encoder import (
    aggregate_layers,
    cos_sim_matrix,
    count_zero_threshold_values,
    flatten_activations_per_sample,
    get_context_groups,
    get_torch_device,
    pick_matching_token_ixs,
    preprocess_activations,
    repackage_flattened_activations,
)
from langbrainscore.utils.logging import log
from langbrainscore.utils.xarray import copy_metadata, fix_xr_dtypes


class HuggingFaceEncoder(_ModelEncoder):
    def __init__(
        self,
        model_id,
        device=None,
        context_dimension: str = None,
        bidirectional: bool = False,
        emb_aggregation: typing.Union[str, None, typing.Callable] = "last",
        emb_preproc: typing.Tuple[str] = (),
        include_special_tokens: bool = True,
    ) -> "HuggingFaceEncoder":

        super().__init__(
            model_id,
            _context_dimension=context_dimension,
            _bidirectional=bidirectional,
            _emb_aggregation=emb_aggregation,
            _emb_preproc=emb_preproc,
            _include_special_tokens=include_special_tokens,
        )

        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.device = device or get_torch_device()
        self.config = AutoConfig.from_pretrained(self._model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self.model = AutoModel.from_pretrained(self._model_id, config=self.config)
        try:
            self.model = self.model.to(self.device)
        except RuntimeError:
            self.device = "cpu"
            self.model = self.model.to(self.device)

    def get_encoder_representations_template(
        self,
    ) -> EncoderRepresentations:
        """
        returns an empty `EncoderRepresentations` object with all the appropriate
        attributes but the `dataset` and `representations` missing and to be filled in
        later.
        """
        return EncoderRepresentations(
            dataset=None,
            representations=xr.DataArray(),  # we don't have these yet
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
        Input a langbrainscore Dataset and return a xarray DataArray of sentence embeddings given the specified
        parameters (in ANNEmbeddingConfig).

        Args:
            dataset (langbrainscore.dataset.DataSet): [description]
            context_dimension (str, optional): the name of a dimension in our xarray-like dataset objcet that provides
                                                groupings of sampleids (stimuli) that should be used
                                                as context when generating encoder representations. for instance, in a word-by-word
                                                presentation paradigm we (may) still want the full sentence as context. [default: None].
            bidirectional (bool, optional): if True, allows using "future" context to generate the representation for a current token
                                            otherwise, only uses what occurs in the "past". some might say, setting this to False
                                            gives you a more biologically plausibly analysis downstream (: [default: False]
            emb_aggregation (typing.Union[str, None, typing.Callable], optional): how to aggregate the hidden states of the encoder
            emb_preproc (typing.Union[list, np.ndarray, None], optional): a list of preprocessing functions to apply to the embeddings.
                        Processing is done layer-wise.

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
                self.get_encoder_representations_template()
            )
            to_check_in_cache.dataset = dataset

            try:
                to_check_in_cache.load_cache()
                return to_check_in_cache
            except FileNotFoundError:
                log(
                    f'unable to load cached representations for "{to_check_in_cache.identifier_string}"',
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
        for group in tqdm(context_groups[np.sort(unique_ixs)]):
            # Mask based on the context group
            mask_context = context_groups == group
            stimuli_in_context = stimuli[mask_context]

            # store model states for each stimulus in this context group
            states_sentences_across_stimuli = []

            ###############################################################################
            # CONTEXT LOOP
            ###############################################################################
            for i, stimulus in enumerate(stimuli_in_context):

                # extract stim to encode based on the uni/bi-directional nature of models
                if not self._bidirectional:
                    stimuli_directional = stimuli_in_context[: i + 1]
                else:
                    stimuli_directional = stimuli_in_context

                # join the stimuli together within a context group
                stimuli_directional = " ".join(
                    map(
                        lambda x: x.strip(), stimuli_directional
                    )  # Strip out odd spaces between stimuli (but *not* within the stimuli).
                )

                tokenized_directional_context = self.tokenizer(
                    stimuli_directional,
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=True,
                ).to(self.device)

                # Get the hidden states
                result_model = self.model(
                    tokenized_directional_context.input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # dict with key=layer, value=3D tensor of dims: [batch, tokens, emb size]
                hidden_states = result_model["hidden_states"]

                layer_wise_activations = dict()

                start_of_interest = stimuli_directional.find(stimulus.strip())
                char_span_of_interest = slice(
                    start_of_interest, start_of_interest + len(stimulus.strip())
                )
                token_span_of_interest = pick_matching_token_ixs(
                    tokenized_directional_context, char_span_of_interest
                )

                all_special_ids = set(self.tokenizer.all_special_ids)
                insert_first_upto = 0
                insert_last_from = tokenized_directional_context.input_ids.shape[-1]
                for i, tid in enumerate(tokenized_directional_context.input_ids[0, :]):
                    if tid.item() in all_special_ids:
                        insert_first_upto = i + 1
                    else:
                        break
                for i in range(
                    1, tokenized_directional_context.input_ids.shape[-1] + 1
                ):
                    tid = tokenized_directional_context.input_ids[0, -i]
                    if tid.item() in all_special_ids:
                        insert_last_from -= 1
                    else:
                        break

                for idx_layer, layer in enumerate(hidden_states):  # Iterate over layers
                    this_extracted = layer[
                        :,  # batch (singleton)
                        token_span_of_interest,  # if self._context_dimension is not None else slice(None), # n_tokens
                        :,  # emb_dim (e.g., 768, 1024, etc)
                    ].squeeze(
                        0
                    )  # collapse batch dim to obtain shape (n_tokens, emb_dim)

                    if self._include_special_tokens:
                        this_extracted = torch.cat(
                            [
                                layer[:, :insert_first_upto, :].squeeze(0),
                                this_extracted,
                            ],
                            axis=0,
                        )
                        this_extracted = torch.cat(
                            [
                                this_extracted,
                                layer[:, insert_last_from:, :].squeeze(0),
                            ],
                            axis=0,
                        )

                    layer_wise_activations[idx_layer] = this_extracted.detach()

                # Aggregate hidden states within a sample
                # states_sentences_agg is a dict with key = layer, value = array of emb dimension
                states_sentences_agg = aggregate_layers(
                    layer_wise_activations, **{"emb_aggregation": self._emb_aggregation}
                )

                # states_sentences_across_stimuli store all the hidden states for the current context group across all stimuli
                states_sentences_across_stimuli.append(states_sentences_agg)

            ###############################################################################
            # END CONTEXT LOOP
            ###############################################################################

            # Flatten activations across layers and package as xarray
            flattened_activations_and_layer_ids = [
                *map(flatten_activations_per_sample, states_sentences_across_stimuli)
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

        ###############################################################################
        # POST-PROCESS ACTIVATIONS
        ###############################################################################
        if len(self._emb_preproc) > 0:  # Preprocess activations
            for p_id in self._emb_preproc:
                activations_2d, layer_ids_1d = preprocess_activations(
                    activations_2d=activations_2d,
                    layer_ids_1d=layer_ids_1d,
                    emb_preproc_mode=p_id,
                )

        assert activations_2d.shape[1] == len(layer_ids_1d)
        assert activations_2d.shape[0] == len(stimuli)

        # Package activations as xarray and reapply metadata
        encoded_dataset = copy_metadata(
            repackage_flattened_activations(
                activations_2d=activations_2d,
                layer_ids_1d=layer_ids_1d,
                dataset=dataset,
            ),
            dataset.contents,
            "sampleid",
        )

        to_return: EncoderRepresentations = self.get_encoder_representations_template()
        to_return.dataset = dataset
        to_return.representations = fix_xr_dtypes(encoded_dataset)

        if write_cache:
            to_return.to_cache(overwrite=True)

        return to_return

    def get_special_token_offset(self) -> int:
        """
        the offset (no. of tokens in tokenized text) from the start to exclude
        when extracting the representation of a particular stimulus. this is
        needed when the stimulus is evaluated in a context group to achieve
        correct boundaries (otherwise we get off-by-context errors)
        """
        with_special_tokens = self.tokenizer("brainscore")["input_ids"]
        first_token_id, *_ = self.tokenizer("brainscore", add_special_tokens=False)[
            "input_ids"
        ]
        special_token_offset = with_special_tokens.index(first_token_id)
        return special_token_offset

    def get_modelcard(self):
        """
        Returns the model card of the model
        NOT DONE!!!
        """

        # Obtain number of layers
        d_config = self.config.to_dict()

        config_specs_of_interest = [
            "n_layer",
            "n_ctx",
            "n_embd",
            "n_head",
            "vocab_size",
        ]

        config_specs = {k: d_config.get(k, None) for k in config_specs_of_interest}

        # Evaluate each layer

    def get_explainable_variance(
        self,
        ann_encoded_dataset,
        method: str = "pca",
        variance_threshold: float = 0.80,
        **kwargs,
    ) -> xr.Dataset:
        """
        Returns how many PCs are needed to explain the variance threshold (default 80%) per layer.

        TODO: move to `langbrainscore.analysis.?` or make @classmethod

        """
        # n_embd = self.config.n_embd

        # Get the PCA explained variance per layer
        layer_ids = ann_encoded_dataset.layer.values
        _, unique_ixs = np.unique(layer_ids, return_index=True)
        # Make sure context group order is preserved
        for layer_id in tqdm(layer_ids[np.sort(unique_ixs)]):
            layer_dataset = (
                ann_encoded_dataset.isel(
                    neuroid=(ann_encoded_dataset.layer == layer_id)
                )
                .drop("timeid")
                .squeeze()
            )
            # assert layer_dataset.shape[1] == n_embd

            # Figure out how many PCs we attempt to fit
            n_comp = np.min([layer_dataset.shape[1], layer_dataset.shape[0]])

            # Get explained variance
            if method == "pca":
                from sklearn.decomposition import PCA

                decomp = PCA(n_components=n_comp)
            elif method == "mds":
                from sklearn.manifold import MDS

                decomp = MDS(n_components=n_comp)
            elif method == "tsne":
                from sklearn.manifold import TSNE

                decomp = TSNE(n_components=n_comp)
            else:
                raise ValueError(f"Unknown method: {method}")

            decomp.fit(layer_dataset.values)
            explained_variance = decomp.explained_variance_ratio_

            # Get the number of PCs needed to explain the variance threshold
            explained_variance_cum = np.cumsum(explained_variance)
            n_pc_needed = np.argmax(explained_variance_cum >= variance_threshold) + 1

            # Store per layer
            layer_id = str(layer_id)
            print(
                f"Layer {layer_id}: {n_pc_needed} PCs needed to explain {variance_threshold} variance"
            )

    def get_layer_sparsity(
        self, ann_encoded_dataset, zero_threshold: float = 0.0001, **kwargs
    ) -> xr.Dataset:
        """
        Check how sparse activations within a given layer are.

        Sparsity is defined as 1 - values below the zero_threshold / total number of values.

        TODO: move to `langbrainscore.analysis.?` or make @classmethod
        """
        # n_embd = self.config.n_embd

        # Get the PCA explained variance per layer
        layer_ids = ann_encoded_dataset.layer.values
        _, unique_ixs = np.unique(layer_ids, return_index=True)
        # Make sure context group order is preserved
        for layer_id in tqdm(layer_ids[np.sort(unique_ixs)]):
            layer_dataset = (
                ann_encoded_dataset.isel(
                    neuroid=(ann_encoded_dataset.layer == layer_id)
                )
                .drop("timeid")
                .squeeze()
            )
            # assert layer_dataset.shape[1] == n_embd

            # Get sparsity
            zero_values = count_zero_threshold_values(
                layer_dataset.values, zero_threshold
            )
            sparsity = 1 - (zero_values / layer_dataset.size)

            # Store per layer
            layer_id = str(layer_id)
            print(f"Layer {layer_id}: {sparsity:.3f} sparsity")

    def get_anisotropy(self, ann_encoded_dataset: EncoderRepresentations):
        raise NotImplementedError


class PTEncoder(_ModelEncoder):
    def __init__(self, model_id: str) -> "PTEncoder":
        super().__init__(model_id)

    def encode(self, dataset: "langbrainscore.dataset.Dataset") -> xr.DataArray:
        # TODO
        pass


class EncoderCheck:
    """
    Class for checking whether obtained embeddings from the Encoder class are correct and similar to other encoder objects.
    """

    def __init__(
        self,
    ):
        pass

    def _load_cached_activations(self, encoded_ann_identifier: str):
        raise NotImplementedError

    def similiarity_metric_across_layers(
        self,
        sim_metric: str = "tol",
        enc1: xr.DataArray = None,
        enc2: xr.DataArray = None,
        tol: float = 1e-8,
        threshold: float = 1e-4,
    ) -> bool:
        """
        Given two activations, iterate across layers and check np.allclose using different tolerance levels.

                Parameters:
            sim_metric: str
                Similarity metric to use.
            enc1: xr.DataArray
                First encoder activations.
            enc2: xr.DataArray
                Second encoder activations.
            tol: float
                Tolerance level to start at (we will iterate upwards the tolerance level). Default is 1e-8.

                        Returns:
                                bool: whether the tolerance level was met (True) or not (False)
                                bad_stim: set of stimuli indices that did not meet tolerance level `threshold` (if any)

        """
        # First check is whether number of layers / shapes match
        assert enc1.shape == enc2.shape
        assert (
            enc1.sampleid.values == enc2.sampleid.values
        ).all()  # ensure that we are looking at the same stimuli
        layer_ids = enc1.layer.values
        _, unique_ixs = np.unique(layer_ids, return_index=True)
        print(f"\n\nChecking similarity across layers using sim_metric: {sim_metric}")

        all_good = True
        bad_stim = set()  # store indices of stimuli that are not similar

        # Iterate across layers
        for layer_id in tqdm(layer_ids[np.sort(unique_ixs)]):
            enc1_layer = enc1.isel(neuroid=(enc1.layer == layer_id))  # .squeeze()
            enc2_layer = enc2.isel(neuroid=(enc2.layer == layer_id))  # .squeeze()

            # Check whether values match. If not, iteratively increase tolerance until values match
            if sim_metric in ("tol", "diff"):
                abs_diff = np.abs(enc1_layer - enc2_layer)
                abs_diff_per_stim = np.max(
                    abs_diff, axis=1
                )  # Obtain the biggest difference aross neuroids (units)
                while (abs_diff_per_stim > tol).all():
                    tol *= 10

            elif "cos" in sim_metric:
                # Check cosine distance between each row, e.g., sentence vector
                cos_sim = cos_sim_matrix(enc1_layer, enc2_layer)
                cos_dist = (
                    1 - cos_sim
                )  # 0 means identical, 1 means orthogonal, 2 means opposite
                # We still want this as close to zero as possible for similar vectors.
                cos_dist_abs = np.abs(cos_dist)
                abs_diff_per_stim = cos_dist_abs

                # Check how close the cosine distance is to 0
                while (cos_dist_abs > tol).all():
                    tol *= 10
            else:
                raise NotImplementedError(f"Invalid `sim_metric`: {sim_metric}")

            print(f"Layer {layer_id}: Similarity at tolerance: {tol:.3e}")
            if tol > threshold:
                print(f"WARNING: Low tolerance level")
                all_good = False
                bad_stim.update(
                    enc1.sampleid[np.where(abs_diff_per_stim > tol)[0]]
                )  # get sampleids of stimuli that are not similar

        return all_good, bad_stim
