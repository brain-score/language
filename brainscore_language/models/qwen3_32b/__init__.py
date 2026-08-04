from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from transformers import AutoConfig

# Qwen3-32B
batch_size = 4
top_k = 0.1 # take 10% of total number of units
model_name = 'Qwen/Qwen3-32B'

config = AutoConfig.from_pretrained(model_name)
num_blocks = config.num_hidden_layers 
hidden_size = config.hidden_size

layer_names = [f'model.layers.{block}' for block in range(num_blocks)]

model_registry['qwen3-32b-loc-10'] = lambda: HuggingfaceSubject(model_id=model_name, 
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: layer_names},
    use_localizer=True,
    localizer_kwargs={
        'hidden_dim': hidden_size,
        'batch_size': batch_size,
        "top_k": top_k,
    }
)