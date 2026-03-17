from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# Qwen2.5-3B: 36 transformer layers, hidden size 2048.
# Layer 35 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['qwen2.5-3b'] = lambda: HuggingfaceSubject(
    model_id='Qwen/Qwen2.5-3B',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'model.layers.35'
    },
)
