from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

# layer assignment based on choosing the maximally scoring layer on Pereira2018-encoding

# LLaMA 1 models

model_registry['llama-7b'] = lambda: HuggingfaceSubject(
    model_id='huggyllama/llama-7b', 
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'model.layers.29.post_attention_layernorm'}
)

model_registry['llama-13b'] = lambda: HuggingfaceSubject(
    model_id='huggyllama/llama-13b', 
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'model.layers.33.post_attention_layernorm'}
)

model_registry['llama-33b'] = lambda: HuggingfaceSubject(
    model_id='huggyllama/llama-30b', 
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'model.layers.53.post_attention_layernorm'}
)

# Alpaca models

model_registry['alpaca-7b'] = lambda: HuggingfaceSubject(
    model_id='chavinlo/alpaca-native', 
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'model.layers.29.post_attention_layernorm'}
)

# Vicuna models

model_registry['vicuna-7b'] = lambda: HuggingfaceSubject(
    model_id='lmsys/vicuna-7b-v1.3', 
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'model.layers.28.post_attention_layernorm'}
)

model_registry['vicuna-13b'] = lambda: HuggingfaceSubject(
    model_id='lmsys/vicuna-13b-v1.3', 
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'model.layers.33.post_attention_layernorm'}
)

model_registry['vicuna-33b'] = lambda: HuggingfaceSubject(
    model_id='lmsys/vicuna-33b-v1.3', 
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'model.layers.52.post_attention_layernorm'}
)
