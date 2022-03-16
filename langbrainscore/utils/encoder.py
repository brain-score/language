import typing

import numpy as np
import torch
import xarray as xr


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


def aggregate_layers(hidden_states: dict, **kwargs):
    """Input a hidden states dictionary (key = layer, value = 2D array of n_tokens x emb_dim)

    Args:
        hidden_states (dict): key = layer (int), value = 2D PyTorch tensor of shape (n_tokens, emb_dim)

    Raises:
        NotImplementedError

    Returns:
        dict: key = layer, value = array of emb_dim
    """
    emb_aggregation = kwargs.get("emb_aggregation")
    states_layers = dict()

    # Iterate over layers
    for i in hidden_states.keys():
        if emb_aggregation == "last":
            state = hidden_states[i][-1, :]  # get last token
        elif emb_aggregation == "mean":
            state = torch.mean(hidden_states[i], dim=0)  # mean over tokens
        elif emb_aggregation == "median":
            state = torch.median(hidden_states[i], dim=0)  # median over tokens
        elif emb_aggregation == "sum":
            state = torch.sum(hidden_states[i], dim=0)  # sum over tokens
        elif emb_aggregation == "all" or emb_aggregation == None:
            state = hidden_states
        else:
            raise NotImplementedError("Sentence embedding method not implemented")

        states_layers[i] = state.detach().numpy()

    return states_layers


def set_case(sample: str = None, emb_case: typing.Union[str, None] = None):
    if emb_case == "lower":
        sample = sample.lower()
    elif emb_case == "upper":
        sample = sample.upper()
    else:
        sample = sample

    return sample


def get_context_groups(dataset, context_dimension):
    if context_dimension is None:
        context_groups = np.arange(0, dataset.stimuli.size, 1)
    else:
        context_groups = dataset.stimuli.coords[context_dimension].values
    return context_groups


def repackage_flattened_activations(
    flattened_activations, states_sentences_agg, layer_ids, dataset
):
    return xr.DataArray(
        np.expand_dims(np.vstack(flattened_activations), axis=2),
        dims=("sampleid", "neuroid", "timeid"),
        coords={
            "sampleid": dataset.contents.sampleid.values,
            "neuroid": np.arange(
                np.sum([len(states_sentences_agg[x]) for x in states_sentences_agg])
            ),
            "timeid": np.arange(1),
            "layer": ("neuroid", np.array(layer_ids[0], dtype="int64")),
        },
    )
