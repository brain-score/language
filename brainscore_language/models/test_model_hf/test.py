"""
Test file for test-model-hf
This is a minimal test to verify the model loads correctly
"""
from brainscore_language import load_model

def test_model_loads():
    """Test that the model can be loaded"""
    model = load_model('test-model-hf')
    assert model is not None
    assert model.identifier() == 'distilgpt2'
    print("✓ Model loads successfully")

if __name__ == '__main__':
    test_model_loads()
