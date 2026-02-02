from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import load_benchmark, ArtificialSubject

model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
layer_name = 'transformer.h.5'

benchmark = load_benchmark('Pereira2018.243sentences-ridge')

layer_model = HuggingfaceSubject(
    model_id='distilgpt2', 
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: layer_name
    }
)

layer_score = benchmark(layer_model)
print(layer_score)
breakpoint()