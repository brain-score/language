from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# Qwen2.5-14B: 48 transformer layers, hidden size 5120.
# Layer 47 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['qwen2.5-14b'] = lambda: HuggingfaceSubject(
    model_id='Qwen/Qwen2.5-14B',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'model.layers.47'
    },
)
