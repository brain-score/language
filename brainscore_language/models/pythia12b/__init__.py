from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# Pythia-12B: 36 transformer layers, hidden size 5120.
# Layer 35 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['pythia-12b'] = lambda: HuggingfaceSubject(
    model_id='EleutherAI/pythia-12b',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'gpt_neox.layers.35'
    },
)
