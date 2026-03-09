from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# Gemma 2B: 18 transformer layers, hidden size 2048.
# Layer 17 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['gemma-2b'] = lambda: HuggingfaceSubject(
    model_id='google/gemma-2b',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'model.layers.17'
    },
)
