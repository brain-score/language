from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# Pythia 2.8B: 32 transformer layers, hidden size 2560.
# Layer 31 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['pythia-2.8b'] = lambda: HuggingfaceSubject(
    model_id='EleutherAI/pythia-2.8b',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'gpt_neox.layers.31'
    },
)
