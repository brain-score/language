from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# Test model using a small Hugging Face model (distilgpt2: ~82M parameters)
# This is a minimal test model for workflow testing
model_registry['test-model-hf'] = lambda: HuggingfaceSubject(
    model_id='distilgpt2',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.5'
    }
)
