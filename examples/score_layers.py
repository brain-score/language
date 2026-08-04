from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import load_benchmark, ArtificialSubject

model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
layer_names = [
    'transformer.h.0',
    'transformer.h.1',
    'transformer.h.2',
    'transformer.h.3',
    'transformer.h.4',
    'transformer.h.5',
]

benchmarks = [
    "Fedorenko2016-ridge",
    "Blank2014-ridge",
    "Pereira2018.384sentences-ridge",
    "Pereira2018.243sentences-ridge"
]

for benchmark_name in benchmarks[2:]:
    benchmark = load_benchmark(benchmark_name)

    layer_model = HuggingfaceSubject(
        model_id='distilgpt2', 
        region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: layer_names
        }
    )

    layer_scores = benchmark(layer_model)
    for layer_name, layer_score in layer_scores.items():
        print(f"Benchmark: {benchmark_name}, Layer: {layer_name} | Score: {layer_score.raw:.4f} (ceiling: {layer_score.ceiling:.4f})")