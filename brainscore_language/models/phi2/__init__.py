from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# Phi-2: 32 transformer layers, hidden size 2560.
# Layer 31 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['phi-2'] = lambda: HuggingfaceSubject(
    model_id='microsoft/phi-2',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'model.layers.31'
    },
)
