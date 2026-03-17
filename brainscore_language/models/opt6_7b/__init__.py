from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# OPT-6.7B: 32 transformer layers, hidden size 4096.
# Layer 31 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['opt-6.7b'] = lambda: HuggingfaceSubject(
    model_id='facebook/opt-6.7b',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'model.decoder.layers.31'
    },
)
