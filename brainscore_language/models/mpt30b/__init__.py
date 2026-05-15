from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# MPT-30B: 48 transformer layers, hidden size 7168.
# Layer 47 (last) chosen as default mapping pending benchmark-driven selection.
model_registry['mpt-30b'] = lambda: HuggingfaceSubject(
    model_id='mosaicml/mpt-30b',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.blocks.47'
    },
)
