import typing

import numpy as np
import torch
import xarray as xr
from tqdm.auto import tqdm
import random

from langbrainscore.utils.preprocessing import preprocessor_classes
from langbrainscore.utils.logging import log


def count_zero_threshold_values(
    A: np.ndarray,
    zero_threshold: float = 0.001,
):
    """Given matrix A, count how many values are below the zero_threshold"""
    return np.sum(A < zero_threshold)


def flatten_activations_per_sample(activations: dict):
    """
    Convert activations into dataframe format

    Args:
        Input (dict): key = layer, value = array of emb_dim

    Returns:
        arr_flat (np.ndarray): 1D ndarray of flattened activations across all layers
        layers_arr (np.ndarray): 1D ndarray of layer indices, corresponding to arr_flat
    """
    layers_arr = []
    arr_flat = []
    for layer, arr in activations.items():  # Iterate over layers
        arr = np.array(arr)
        arr_flat.append(arr)
        for i in range(arr.shape[0]):  # Iterate across units
            layers_arr.append(layer)
    arr_flat = np.concatenate(
        arr_flat, axis=0
    )  # concatenated activations across layers

    return arr_flat, np.array(layers_arr)


def aggregate_layers(
    hidden_states: dict, mode: typing.Union[str, typing.Callable]
) -> np.ndarray:
    """Input a hidden states dictionary (key = layer, value = 2D array of n_tokens x emb_dim)

    Args:
        hidden_states (dict): key = layer (int), value = 2D PyTorch tensor of shape (n_tokens, emb_dim)

    Raises:
        NotImplementedError

    Returns:
        dict: key = layer, value = array of emb_dim
    """
    states_layers = dict()

    emb_aggregation = mode
    # iterate over layers
    for i in hidden_states.keys():
        if emb_aggregation == "last":
            state = hidden_states[i][-1, :]  # get last token
        elif emb_aggregation == "first":
            state = hidden_states[i][0, :]  # get first token
        elif emb_aggregation == "mean":
            state = torch.mean(hidden_states[i], dim=0)  # mean over tokens
        elif emb_aggregation == "median":
            state = torch.median(hidden_states[i], dim=0)  # median over tokens
        elif emb_aggregation == "sum":
            state = torch.sum(hidden_states[i], dim=0)  # sum over tokens
        elif emb_aggregation == "all" or emb_aggregation == None:
            state = hidden_states
        elif callable(emb_aggregation):
            state = emb_aggregation(hidden_states[i])
        else:
            raise NotImplementedError(
                f"Sentence embedding method [{emb_aggregation}] not implemented"
            )

        states_layers[i] = state.detach().cpu().numpy()

    return states_layers


def get_torch_device():
    """
    get torch device based on whether cuda is available or not
    """
    import torch

    # Set device to GPU if cuda is available.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = torch.device("cpu")
    return device


def set_case(sample: str, emb_case: typing.Union[str, None] = None):
    if emb_case == "lower":
        return sample.lower()
    elif emb_case == "upper":
        return sample.upper()
    return sample


def get_context_groups(dataset, context_dimension):
    if context_dimension is None:
        context_groups = np.arange(0, dataset.stimuli.size, 1)
    else:
        context_groups = dataset.stimuli.coords[context_dimension].values
    return context_groups


def preprocess_activations(*args, **kwargs):
    return postprocess_activations(*args, **kwargs)


def postprocess_activations(
    activations_2d: np.ndarray = None,
    layer_ids_1d: np.ndarray = None,
    emb_preproc_mode: str = None,  # "demean",
):

    activations_processed = []
    layer_ids_processed = []

    # log(f"Preprocessing activations with {p_id}")
    for l_id in np.sort(np.unique(layer_ids_1d)):  # For each layer
        preprocessor = preprocessor_classes[emb_preproc_mode]

        # Get the activations for this layer and retain 2d shape: [n_samples, emb_dim]
        activations_2d_layer = activations_2d[:, layer_ids_1d == l_id]

        preprocessor.fit(
            activations_2d_layer
        )  # obtain a scaling per unit (in emb space)

        # Apply the scaling to the activations and reassamble the activations (might have different shape than original)
        activations_2d_layer_processed = preprocessor.transform(activations_2d_layer)
        activations_processed += [activations_2d_layer_processed]
        layer_ids_processed += [np.full(activations_2d_layer_processed.shape[1], l_id)]

    # Concatenate to obtain [n_samples, emb_dim across layers], i.e., flattened activations
    activations_2d_layer_processed = np.hstack(activations_processed)
    layer_ids_1d_processed = np.hstack(layer_ids_processed)

    return activations_2d_layer_processed, layer_ids_1d_processed


def repackage_flattened_activations(
    activations_2d: np.ndarray = None,
    layer_ids_1d: np.ndarray = None,
    dataset: xr.Dataset = None,
):
    return xr.DataArray(
        np.expand_dims(activations_2d, axis=2),  # add in time dimension
        dims=("sampleid", "neuroid", "timeid"),
        coords={
            "sampleid": dataset.contents.sampleid.values,
            "neuroid": np.arange(len(layer_ids_1d)),
            "timeid": np.arange(1),
            "layer": ("neuroid", np.array(layer_ids_1d, dtype="int64")),
        },
    )


def cos_sim_matrix(A, B):
    """Compute the cosine similarity matrix between two matrices A and B.
    1 means the two vectors are identical. 0 means they are orthogonal.
    -1 means they are opposite."""
    return (A * B).sum(axis=1) / (A * A).sum(axis=1) ** 0.5 / (B * B).sum(axis=1) ** 0.5


def pick_matching_token_ixs(
    batchencoding: "transformers.tokenization_utils_base.BatchEncoding",
    char_span_of_interest: slice,
) -> slice:
    """Picks token indices in a tokenized encoded sequence that best correspond to
        a substring of interest in the original sequence, given by a char span (slice)

    Args:
        batchencoding (transformers.tokenization_utils_base.BatchEncoding): the output of a
            `tokenizer(text)` call on a single text instance (not a batch, i.e. `tokenizer([text])`).
        char_span_of_interest (slice): a `slice` object denoting the character indices in the
            original `text` string we want to extract the corresponding tokens for

    Returns:
        slice: the start and stop indices within an encoded sequence that
            best match the `char_span_of_interest`
    """
    from transformers import tokenization_utils_base

    start_token = 0
    end_token = batchencoding.input_ids.shape[-1]
    for i, _ in enumerate(batchencoding.input_ids.reshape(-1)):
        span = batchencoding[0].token_to_chars(
            i
        )  # batchencoding 0 gives access to the encoded string

        if span is None:  # for [CLS], no span is returned
            log(
                f'No span returned for token at {i}: "{batchencoding.tokens()[i]}"',
                type="WARN",
                cmap="WARN",
                verbosity_check=True,
            )
            continue
        else:
            span = tokenization_utils_base.CharSpan(*span)

        if span.start <= char_span_of_interest.start:
            start_token = i
        if span.end >= char_span_of_interest.stop:
            end_token = i + 1
            break

    assert (
        end_token - start_token <= batchencoding.input_ids.shape[-1]
    ), f"Extracted span is larger than original span"

    return slice(start_token, end_token)


def encode_stimuli_in_context(
    stimuli_in_context,
    tokenizer: "transformers.AutoTokenizer",
    model: "transformers.AutoModel",
    bidirectional: bool,
    include_special_tokens: bool,
    emb_aggregation,
    device=get_torch_device(),
):
    """ """
    # CONTEXT LOOP
    for i, stimulus in enumerate(stimuli_in_context):

        # extract stim to encode based on the uni/bi-directional nature of models
        if not bidirectional:
            stimuli_directional = stimuli_in_context[: i + 1]
        else:
            stimuli_directional = stimuli_in_context

        # join the stimuli together within a context group using just a single space
        stimuli_directional = " ".join(stimuli_directional)

        tokenized_directional_context = tokenizer(
            stimuli_directional,
            padding=False,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)

        # Get the hidden states
        result_model = model(
            tokenized_directional_context.input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        # dict with key=layer, value=3D tensor of dims: [batch, tokens, emb size]
        hidden_states = result_model["hidden_states"]

        layer_wise_activations = dict()

        # Find which indices match the current stimulus in the given context group
        start_of_interest = stimuli_directional.find(stimulus)
        char_span_of_interest = slice(
            start_of_interest, start_of_interest + len(stimulus)
        )
        token_span_of_interest = pick_matching_token_ixs(
            tokenized_directional_context, char_span_of_interest
        )

        log(
            f"Interested in the following stimulus:\n{stimuli_directional[char_span_of_interest]}\n"
            f"Recovered:\n{tokenized_directional_context.tokens()[token_span_of_interest]}",
            cmap="INFO",
            type="INFO",
            verbosity_check=True,
        )

        all_special_ids = set(tokenizer.all_special_ids)

        # Look for special tokens in the beginning and end of the sequence
        insert_first_upto = 0
        insert_last_from = tokenized_directional_context.input_ids.shape[-1]
        # loop through input ids
        for i, tid in enumerate(tokenized_directional_context.input_ids[0, :]):
            if tid.item() in all_special_ids:
                insert_first_upto = i + 1
            else:
                break
        for i in range(1, tokenized_directional_context.input_ids.shape[-1] + 1):
            tid = tokenized_directional_context.input_ids[0, -i]
            if tid.item() in all_special_ids:
                insert_last_from -= 1
            else:
                break

        for idx_layer, layer in enumerate(hidden_states):  # Iterate over layers
            # b (1), n (tokens), h (768, ...)
            # collapse batch dim to obtain shape (n_tokens, emb_dim)
            this_extracted = layer[
                :,
                token_span_of_interest,
                :,
            ].squeeze(0)

            if include_special_tokens:
                # get the embeddings for the first special tokens
                this_extracted = torch.cat(
                    [
                        layer[:, :insert_first_upto, :].squeeze(0),
                        this_extracted,
                    ],
                    axis=0,
                )
                # get the embeddings for the last special tokens
                this_extracted = torch.cat(
                    [
                        this_extracted,
                        layer[:, insert_last_from:, :].squeeze(0),
                    ],
                    axis=0,
                )

            layer_wise_activations[idx_layer] = this_extracted.detach()

        # Aggregate hidden states within a sample
        # aggregated_layerwise_sentence_encodings is a dict with key = layer, value = array of emb dimension
        aggregated_layerwise_sentence_encodings = aggregate_layers(
            layer_wise_activations, mode=emb_aggregation
        )
        yield aggregated_layerwise_sentence_encodings
    # END CONTEXT LOOP


def dataset_from_stimuli(stimuli: "pd.DataFrame"):
    pass


###############################################################################
# ANALYSIS UTILS: these act upon encoded data, rather than encoders
###############################################################################


def get_decomposition_method(method: str = "pca", n_comp: int = 10, **kwargs):
    """
    Return the sklearn method to use for decomposition.

    Args:
            method (str): Method to use for decomposition (default: "pca", other options: "mds", "tsne")
            n_comp (int): Number of components to keep (default: 10)

    Returns:
            sklearn method
    """

    if method == "pca":
        from sklearn.decomposition import PCA

        decomp_method = PCA(n_components=n_comp)

    elif method == "mds":
        from sklearn.manifold import MDS

        decomp_method = MDS(n_components=n_comp)

    elif method == "tsne":
        from sklearn.manifold import TSNE

        decomp_method = TSNE(n_components=n_comp)

    else:
        raise ValueError(f"Unknown method: {method}")

    return decomp_method


def get_explainable_variance(
    ann_encoded_dataset,
    method: str = "pca",
    variance_threshold: float = 0.80,
    **kwargs,
) -> xr.Dataset:
    """
    Returns how many components are needed to explain the variance threshold (default 80%) per layer.

    Args:
            ann_encoded_dataset (xr.Dataset): ANN encoded dataset
            method (str): Method to use for decomposition (default: "pca", other options: "mds", "tsne")
            variance_threshold (float): Variance threshold to use for determining how many components are needed to
                    explain explained a certain threshold of variance (default: 0.80)
            **kwargs: Additional keyword arguments to pass to the underlying method

    Returns:
            variance_across_layers (dict): Nested dict with value of interest as key (e.g., explained variance) and
                    layer id as key (e.g., 0, 1, 2, ...) with corresponding values.

    """

    ks = [
        f"n_comp-{method}_needed-{variance_threshold}",
        f"first_comp-{method}_explained_variance",
    ]
    variance_across_layers = {k: {} for k in ks}

    # Get the PCA explained variance per layer
    layer_ids = ann_encoded_dataset.layer.values
    _, unique_ixs = np.unique(layer_ids, return_index=True)

    # Make sure that layer order is preserved
    for layer_id in tqdm(layer_ids[np.sort(unique_ixs)]):
        layer_dataset = (
            ann_encoded_dataset.isel(neuroid=(ann_encoded_dataset.layer == layer_id))
            .drop("timeid")
            .squeeze()
        )

        # Figure out how many PCs we attempt to fit
        n_comp = np.min([layer_dataset.shape[1], layer_dataset.shape[0]])

        # Get explained variance
        decomp_method = get_decomposition_method(method=method, n_comp=n_comp, **kwargs)

        decomp_method.fit(layer_dataset.values)
        explained_variance = decomp_method.explained_variance_ratio_

        # Get the number of PCs needed to explain the variance threshold
        explained_variance_cum = np.cumsum(explained_variance)
        n_pc_needed = np.argmax(explained_variance_cum >= variance_threshold) + 1

        # Store per layer
        layer_id = str(layer_id)
        print(
            f"Layer {layer_id}: {n_pc_needed} PCs needed to explain {variance_threshold} variance "
            f"with the 1st PC explaining {explained_variance[0]:.2f}% of the total variance"
        )

        variance_across_layers[f"n_comp-{method}_needed-{variance_threshold}"][
            layer_id
        ] = n_pc_needed
        variance_across_layers[f"first_comp-{method}_explained_variance"][
            layer_id
        ] = explained_variance[0]

    return variance_across_layers


def get_layer_sparsity(
    ann_encoded_dataset, zero_threshold: float = 0.0001, **kwargs
) -> xr.Dataset:
    """
    Check how sparse activations within a given layer are.

    Sparsity is defined as 1 - values below the zero_threshold / total number of values.

    Args:
            ann_encoded_dataset (xr.Dataset): ANN encoded dataset
            zero_threshold (float): Threshold to use for determining sparsity (default: 0.0001)
            **kwargs: Additional keyword arguments to pass to the underlying method

    Returns:
            sparsity_across_layers (dict): Nested dict with value of interest as key (e.g., sparsity) and
                    layer id as key (e.g., 0, 1, 2, ...) with corresponding values.

    """
    # Obtain embedding dimension (for sanity checks)
    # if self.model_specs["hidden_emb_dim"]:
    #     hidden_emb_dim = self.model_specs["hidden_emb_dim"]
    # else:
    #     hidden_emb_dim = None
    #     log(
    #         f"Hidden embedding dimension not specified yet",
    #         cmap="WARN",
    #         type="WARN",
    #     )

    ks = [f"sparsity-{zero_threshold}"]
    sparsity_across_layers = {k: {} for k in ks}

    # Get the PCA explained variance per layer
    layer_ids = ann_encoded_dataset.layer.values
    _, unique_ixs = np.unique(layer_ids, return_index=True)

    # Make sure that layer order is preserved
    for layer_id in tqdm(layer_ids[np.sort(unique_ixs)]):
        layer_dataset = (
            ann_encoded_dataset.isel(neuroid=(ann_encoded_dataset.layer == layer_id))
            .drop("timeid")
            .squeeze()
        )

        # if hidden_emb_dim is not None:
        #     assert layer_dataset.shape[1] == hidden_emb_dim
        #
        # Get sparsity
        zero_values = count_zero_threshold_values(layer_dataset.values, zero_threshold)
        sparsity = 1 - (zero_values / layer_dataset.size)

        # Store per layer
        layer_id = str(layer_id)
        print(f"Layer {layer_id}: {sparsity:.3f} sparsity")

        sparsity_across_layers[f"sparsity-{zero_threshold}"][layer_id] = sparsity

    return sparsity_across_layers


def cos_contrib(
    emb1: np.ndarray,
    emb2: np.ndarray,
):
    """
    Cosine contribution function defined in eq. 3 by Timkey & van Schijndel (2021): https://arxiv.org/abs/2109.04404

    Args:
        emb1 (np.ndarray): Embedding vector 1
        emb2 (np.ndarray): Embedding vector 2

    Returns:
        cos_contrib (float): Cosine contribution

    """

    numerator_terms = emb1 * emb2
    denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    return numerator_terms / denom


def get_anisotropy(
    ann_encoded_dataset: "EncoderRepresentations", num_random_samples: int = 1000
):
    """
    Calculate the anisotropy of the embedding vectors as Timkey & van Schijndel (2021): https://arxiv.org/abs/2109.04404
    (base function from their GitHub repo: https://github.com/wtimkey/rogue-dimensions/blob/main/replication.ipynb,
    but modified to work within the Language Brain-Score project)


    """
    rogue_dist = []
    num_toks = len(ann_encoded_dataset.sampleid)  # Number of stimuli

    # randomly sample embedding pairs to compute avg. cosine similiarity contribution
    random_pairs = [
        random.sample(range(num_toks), 2) for i in range(num_random_samples)
    ]

    cos_contribs_by_layer = []

    layer_ids = ann_encoded_dataset.layer.values
    _, unique_ixs = np.unique(layer_ids, return_index=True)

    for layer_id in tqdm(layer_ids[np.sort(unique_ixs)]):
        layer_dataset = (
            ann_encoded_dataset.isel(neuroid=(ann_encoded_dataset.layer == layer_id))
            .drop("timeid")
            .squeeze()
        )

        layer_cosine_contribs = []
        layer_rogue_cos_contribs = []
        for pair in random_pairs:
            emb1 = sample_data[layer, pair[0], :]  # fix
            emb2 = sample_data[layer, pair[1], :]
            layer_cosine_contribs.append(cos_contrib(emb1, emb2))

        layer_cosine_contribs = np.array(layer_cosine_contribs)
        layer_cosine_sims = layer_cosine_contribs.sum(axis=1)
        layer_cosine_contribs_mean = layer_cosine_contribs.mean(axis=0)
        cos_contribs_by_layer.append(layer_cosine_contribs_mean)
    cos_contribs_by_layer = np.array(cos_contribs_by_layer)

    aniso = cos_contribs_by_layer.sum(
        axis=1
    )  # total anisotropy, measured as avg. cosine sim between random emb. pairs

    for layer in range(num_layers[model_name]):
        top_3_dims = np.argsort(cos_contribs_by_layer[layer])[-3:]
        top = cos_contribs_by_layer[layer, top_3_dims[2]] / aniso[layer]
        second = cos_contribs_by_layer[layer, top_3_dims[1]] / aniso[layer]
        third = cos_contribs_by_layer[layer, top_3_dims[0]] / aniso[layer]
        print(
            "& {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                layer, top, second, third, aniso[layer]
            )
        )
