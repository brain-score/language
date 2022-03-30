import typing

import numpy as np
import torch
import xarray as xr

from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer, RobustScaler
from sklearn.decomposition import PCA

preprocessor_classes = {
    'demean': StandardScaler(with_std=False),
    'demean_std': StandardScaler(with_std=True),
    'minmax': MinMaxScaler,
    'discretize': KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'),
    'robust_scaler': RobustScaler(),
    'pca': PCA(n_components=10),
}

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


def get_torch_device():
    '''
    get torch device based on whether cuda is available or not
    '''
    import torch
    # Set device to GPU if cuda is available.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = torch.device("cpu")
    return device


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

def preprocess_activations(activations_2d: np.ndarray = None,
                           layer_ids_1d: np.ndarray = None,
                           emb_preproc: typing.Union[list, np.ndarray] = ['demean'],):
    
    activations_processed = []
    layer_ids_processed = []
    
    for p_id in emb_preproc:  # For each preprocessing setting
        # log(f"Preprocessing activations with {p_id}")
        for l_id in np.sort(np.unique(layer_ids_1d)):  # For each layer
            preprocessor = preprocessor_classes[p_id]
            
            # Get the activations for this layer and retain 2d shape: [n_samples, emb_dim]
            activations_2d_layer = activations_2d[:, layer_ids_1d == l_id]
            
            preprocessor.fit(activations_2d_layer)  # obtain a scaling per unit (in emb space)
            
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
    dataset: xr.Dataset = None):
    return xr.DataArray(
        np.expand_dims(activations_2d, axis=2), # add in time dimension
        dims=("sampleid", "neuroid", "timeid"),
        coords={
            "sampleid": dataset.contents.sampleid.values,
            "neuroid": np.arange(len(layer_ids_1d)),
            "timeid": np.arange(1),
            "layer": ("neuroid", np.array(layer_ids_1d, dtype="int64")),})

