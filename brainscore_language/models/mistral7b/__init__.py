from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# Mistral-7B-v0.1: 32 transformer layers, hidden size 4096.
# Layer 31 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['mistral-7b'] = lambda: HuggingfaceSubject(
    model_id='mistralai/Mistral-7B-v0.1',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'model.layers.31'
    },
)
