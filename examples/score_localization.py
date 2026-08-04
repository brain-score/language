from brainscore_language import load_benchmark
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import ArtificialSubject
from transformers import AutoConfig

def score_hf_model(model_name: str, benchmark_name: str, batch_size: int=16, top_k: int|float=0.1):
    config = AutoConfig.from_pretrained(model_name)
    num_blocks = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layer
    hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else config.d_model

    layer_names = [f'model.layers.{block}' for block in range(num_blocks)]

    model = HuggingfaceSubject(model_id=model_name, 
        region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: layer_names},
        use_localizer=True,
        localizer_kwargs={
            'hidden_dim': hidden_size,
            'batch_size': batch_size,
            "top_k": top_k,
        }
    )

    benchmark = load_benchmark(benchmark_name)

    model_score = benchmark(model)

    return model_score

if __name__ == '__main__':

    model_name = 'Qwen/Qwen3-4B'
    benchmark_name = 'Pereira2018.243sentences-ridge'
    top_k = 0.1 # take 10% of total number of units
    batch_size = 16

    model_score = score_hf_model(model_name, benchmark_name, batch_size, top_k)

    print(model_score)