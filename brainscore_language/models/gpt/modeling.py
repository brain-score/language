import numpy as np
import torch
from collections import OrderedDict
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from transformers import AutoModelForCausalLM
import re

def get_layer_names(model_id):
    model_ = AutoModelForCausalLM.from_pretrained(model_id)
    modul_names = [x[0] for x in model_.named_modules()]
    layer_drop_names = [x for x in modul_names if len(re.findall(r'.drop$', x)) > 0]
    layer_output_names = [x for x in modul_names if len(re.findall(r'h.\d+$', x)) > 0]
    flat_list = [item for sublist in [layer_drop_names,layer_output_names] for item in sublist]
    return flat_list