import typing

import numpy as np
import torch
import xarray as xr

# from nltk import edit_distance

from langbrainscore.utils.resources import preprocessor_classes
from langbrainscore.utils.logging import log, get_verbosity


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
            raise NotImplementedError("Sentence embedding method not implemented")

        states_layers[i] = state.detach().numpy()

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


def postprocess_activations(
    activations_2d: np.ndarray = None,
    layer_ids_1d: np.ndarray = None,
    emb_preproc_mode: str = "demean",
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
            if get_verbosity():
                log(
                    f'No span returned for token at {i}: "{batchencoding.tokens()[i]}"',
                    type="WARN",
                    cmap="WARN",
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
    ###############################################################################
    # CONTEXT LOOP
    ###############################################################################
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

        if get_verbosity():
            log(
                f"Interested in the following stimulus:\n{stimuli_directional[char_span_of_interest]}\n"
                f"Recovered:\n{tokenized_directional_context.tokens()[token_span_of_interest]}",
                cmap="INFO",
                type="INFO",
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

    ###############################################################################
    # END CONTEXT LOOP
    ###############################################################################


# def get_index(tokenizer, supstr_tokens, substr, mode):
#     supstr_tokens = list(supstr_tokens.squeeze())
#     assert mode in ["start", "stop"]
#     edit_distances = []
#     for idx in range(len(supstr_tokens) + 1):
#         if mode == "start":
#             candidate_tokens = supstr_tokens[idx:]
#         else:
#             candidate_tokens = supstr_tokens[:idx]
#         candidate = tokenizer.decode(candidate_tokens)
#         if mode == "start":
#             comp = candidate[: len(substr)]
#         else:
#             comp = candidate[-len(substr) :]
#         edit_distances.append(edit_distance(comp, substr))
#     return np.argmin(edit_distances)
