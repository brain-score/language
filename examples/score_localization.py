from tqdm import tqdm
from brainscore_language import load_benchmark
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import ArtificialSubject

benchmark = load_benchmark('Pereira2018.243sentences-linear')

num_blocks = 12
layer_names = [f'transformer.h.{block}.{layer_type}' 
    for block in range(num_blocks) 
    for layer_type in ['ln_1', 'attn', 'ln_2', 'mlp']
]

model = HuggingfaceSubject(model_id='gpt2', 
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: layer_names},
    use_localizer=True,
    localizer_kwargs={
        'hidden_dim': 768,
        'batch_size': 16,
        "top_k": 4096,
    }
)

model_score = benchmark(model)

print(model_score)