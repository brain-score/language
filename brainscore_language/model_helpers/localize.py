from typing import List
from collections import OrderedDict

import os
import scipy
import torch
import logging
import numpy as np
import transformers
import pandas as pd

from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from brainscore_language import load_dataset

# To cache the language mask
BRAINIO_CACHE = os.environ.get("BRAINIO", f"{Path.home()}/.brainio")

logger = logging.getLogger(__name__)

# Code adapted from: https://github.com/bkhmsi/brain-language-suma

class Fed10_langlocDataset(Dataset):
    def __init__(self):
        self.num_samples = 240

        data = load_dataset("Fedorenko2010.localization")
        self.sentences = data[data["stim14"]=="S"]["sent"]
        self.non_words = data[data["stim14"]=="N"]["sent"]

    def __getitem__(self, idx):
        return self.sentences.iloc[idx].strip(), self.non_words.iloc[idx].strip()
    
    def __len__(self):
        return len(self.sentences)

def _get_layer(module, layer_name: str) -> torch.nn.Module:
    SUBMODULE_SEPARATOR = '.'
    for part in layer_name.split(SUBMODULE_SEPARATOR):
        module = module._modules.get(part)
        assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
    return module
    
def _register_hook(layer: torch.nn.Module,
                    key: str,
                    target_dict: dict):
    # instantiate parameters to function defaults; otherwise they would change on next function call
    def hook_function(_layer: torch.nn.Module, _input, output: torch.Tensor, key=key):
        # fix for when taking out only the hidden state, this is different from dropout because of residual state
        # see:  https://github.com/huggingface/transformers/blob/c06d55564740ebdaaf866ffbbbabf8843b34df4b/src/transformers/models/gpt2/modeling_gpt2.py#L428
        output = output[0] if isinstance(output, (tuple, list)) else output
        target_dict[key] = output

    hook = layer.register_forward_hook(hook_function)
    return hook

def setup_hooks(model, layer_names):
    """ set up the hooks for recording internal neural activity from the model (aka layer activations) """
    hooks = []
    layer_representations = OrderedDict()

    for layer_name in layer_names:
        layer = _get_layer(model, layer_name)
        hook = _register_hook(layer, key=layer_name,
                                target_dict=layer_representations)
        hooks.append(hook)

    return hooks, layer_representations

def extract_batch(
    model: torch.nn.Module, 
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor,
    layer_names: List[str],
):
    
    batch_activations = {layer_name: [] for layer_name in layer_names}
    hooks, layer_representations = setup_hooks(model, layer_names)

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    for sample_idx in range(len(input_ids)):
        for layer_idx, layer_name in enumerate(layer_names):
            activations = layer_representations[layer_name][sample_idx][-1].cpu()    
            batch_activations[layer_name] += [activations]

    for hook in hooks:
        hook.remove()

    return batch_activations

def extract_representations(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    layer_names: List[str],
    hidden_dim: int,
    batch_size: int,
    device: torch.device,
):
    langloc_dataset = Fed10_langlocDataset()

    # Get the activations of the model on the dataset
    langloc_dataloader = DataLoader(langloc_dataset, batch_size=batch_size, num_workers=0)

    logger.debug(f"> Using Device: {device}")

    model.eval()
    model.to(device)

    final_layer_representations = {
        "sentences": {layer_name: np.zeros((langloc_dataset.num_samples, hidden_dim)) for layer_name in layer_names},
        "non-words": {layer_name: np.zeros((langloc_dataset.num_samples, hidden_dim)) for layer_name in layer_names}
    }
    
    for batch_idx, batch_data in tqdm(enumerate(langloc_dataloader)):

        sents, non_words = batch_data
        sent_tokens = tokenizer(sents, truncation=True, max_length=12, return_tensors='pt').to(device)
        non_words_tokens = tokenizer(non_words, truncation=True, max_length=12, return_tensors='pt').to(device)
        assert sent_tokens.input_ids.size(1) == non_words_tokens.input_ids.size(1)
        
        batch_real_actv = extract_batch(model, sent_tokens["input_ids"], sent_tokens["attention_mask"], layer_names)
        batch_rand_actv = extract_batch(model, non_words_tokens["input_ids"], non_words_tokens["attention_mask"], layer_names)

        for layer_name in layer_names:
            final_layer_representations["sentences"][layer_name][batch_idx*batch_size:(batch_idx+1)*batch_size] = torch.stack(batch_real_actv[layer_name]).numpy()
            final_layer_representations["non-words"][layer_name][batch_idx*batch_size:(batch_idx+1)*batch_size] = torch.stack(batch_rand_actv[layer_name]).numpy()

    return final_layer_representations

def localize_fed10(model_id: str,
    model: torch.nn.Module, 
    top_k: int, 
    tokenizer: transformers.PreTrainedTokenizer, 
    hidden_dim: int, 
    layer_names: List[str], 
    batch_size: int,
    device: torch.device,
):
    """
    Localize the model by selecting the top `top_k` units.
    """

    save_path = f"{BRAINIO_CACHE}/{model_id}_language_mask.npy"

    if os.path.exists(save_path):
        logger.debug(f"Loading language mask from {save_path}")
        return np.load(save_path)

    representations = extract_representations(model, tokenizer, layer_names, hidden_dim, batch_size, device)

    p_values_matrix = np.zeros((len(layer_names), hidden_dim))
    t_values_matrix = np.zeros((len(layer_names), hidden_dim))

    for layer_idx, layer_name in tqdm(enumerate(layer_names)):

        sentences_actv = representations["sentences"][layer_name]
        non_words_actv = representations["non-words"][layer_name]

        t_values_matrix[layer_idx], p_values_matrix[layer_idx] = scipy.stats.ttest_ind(sentences_actv, non_words_actv, axis=0, equal_var=False)
 
    def is_topk(a, k=1):
        _, rix = np.unique(-a, return_inverse=True)
        return np.where(rix < k, 1, 0).reshape(a.shape)

    language_mask = is_topk(t_values_matrix, k=top_k)

    np.save(save_path, language_mask)
    logger.debug(f"{model_id} language mask cached to {save_path}")
    return language_mask
