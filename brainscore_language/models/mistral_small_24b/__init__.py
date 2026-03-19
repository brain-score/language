from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# Mistral-Small-24B: 40 transformer layers, hidden size 5120.
# Layer 39 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['mistral-small-24b'] = lambda: HuggingfaceSubject(
    model_id='mistralai/Mistral-Small-24B',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'model.layers.39'
    },
)
