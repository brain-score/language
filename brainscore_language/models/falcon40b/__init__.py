from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# Falcon-40B: 60 transformer layers, hidden size 8192.
# Layer 59 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['falcon-40b'] = lambda: HuggingfaceSubject(
    model_id='tiiuae/falcon-40b',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.59'
    },
)
