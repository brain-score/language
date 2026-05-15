from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# OPT-30B: 48 transformer layers, hidden size 7168.
# Layer 47 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['opt-30b'] = lambda: HuggingfaceSubject(
    model_id='facebook/opt-30b',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'model.decoder.layers.47'
    },
)
