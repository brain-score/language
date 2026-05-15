from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# Falcon-7B: 32 transformer layers, hidden size 4544.
# Layer 31 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['falcon-7b'] = lambda: HuggingfaceSubject(
    model_id='tiiuae/falcon-7b',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.31'
    },
)
