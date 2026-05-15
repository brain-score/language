from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# TinyLlama 1.1B: 22 transformer layers, hidden size 2048.
# Layer 21 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['tinyllama-1.1b'] = lambda: HuggingfaceSubject(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'model.layers.21'
    },
)
