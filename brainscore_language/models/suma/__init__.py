from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language.model_helpers.modeling_suma import SUMAModel, SUMAConfig
from transformers import AutoTokenizer

layer_names = [f'layers.{layer_num}.{layer_desc}' 
    for layer_num in range(1) 
    for layer_desc in ["input_layernorm", "self_attn"]
]

model_registry['suma'] = lambda: HuggingfaceSubject(
    model_id='suma', 
    model=SUMAModel(
        config=SUMAConfig(
            num_hidden_layers=1,
            num_attention_heads=512,
            num_key_value_heads=512,
            num_cycles=2,
        )
    ), 
    tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf'),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: layer_names},
    use_localizer=True,
    localizer_kwargs={
        'hidden_dim': 4096,
        'batch_size': 16,
        "top_k": 4096,
    }
)