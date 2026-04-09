"""
Test that KV-cached neural extraction produces identical activations
to a full-recompute forward pass. This validates the optimization in
huggingface.py that enables use_cache=True for all modes (including
neural recording), not just behavioral tasks.
"""

import numpy as np
import pytest
import torch

from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language.model_helpers.preprocessing import prepare_context


@pytest.fixture(scope='module')
def subject():
    """Create a distilgpt2 subject for testing. Module-scoped to avoid repeated loading."""
    layer_name = 'transformer.h.0'
    s = HuggingfaceSubject(
        model_id='distilgpt2',
        region_layer_mapping={'language': layer_name},
    )
    return s, layer_name


class TestKVCacheNeuralActivations:

    def test_kv_cached_activations_match_full_recompute(self, subject):
        """
        Run digest_text with KV cache (the default) and compare the final
        word's neural activation against a single full-context forward pass.
        They must match because KV-cached attention at position i depends on
        the same positions 1..i as a full recompute.
        """
        s, layer_name = subject

        # Run digest_text with neural recording (uses KV cache internally)
        s.neural_recordings = []
        s.behavioral_task = None
        s.start_neural_recording('language', ArtificialSubject.RecordingType.fMRI)

        text = ["The", "quick", "brown", "fox"]
        result = s.digest_text(text)
        neural = result['neural']

        # Get the last presentation's activation (for "fox" with full context "The quick brown fox")
        kv_activation = neural.values[-1]  # shape: (n_neuroids,)

        # Full-recompute comparison: single forward pass on full context
        full_context = prepare_context(text)
        tokens = s.tokenizer(full_context, return_tensors='pt').to(s.device)

        layer_activations = {}
        layer = s._get_layer(layer_name)

        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            layer_activations['full'] = out

        hook = layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            s.basemodel(**tokens)
        hook.remove()

        full_activation = layer_activations['full'][:, -1, :].squeeze(0).cpu().numpy()

        np.testing.assert_allclose(
            kv_activation, full_activation,
            rtol=1e-4, atol=1e-5,
            err_msg="KV-cached activation diverges from full-recompute activation"
        )

    def test_kv_cache_produces_valid_neural_output(self, subject):
        """Basic sanity: neural output has correct shape and non-zero values."""
        s, layer_name = subject

        s.neural_recordings = []
        s.behavioral_task = None
        s.start_neural_recording('language', ArtificialSubject.RecordingType.fMRI)

        text = ["Hello", "world"]
        result = s.digest_text(text)
        neural = result['neural']

        assert neural.dims == ('presentation', 'neuroid')
        assert neural.shape[0] == 2  # two presentations
        assert neural.shape[1] > 0  # has neurons
        assert not np.all(neural.values == 0)  # non-trivial activations

    def test_kv_cache_behavioral_still_works(self, subject):
        """Behavioral output (next word prediction) still works with KV cache."""
        s, _ = subject

        s.neural_recordings = []
        s.behavioral_task = None
        s.start_behavioral_task(ArtificialSubject.Task.next_word)

        text = ["The", "quick", "brown"]
        result = s.digest_text(text)
        behavior = result['behavior']

        assert behavior.dims == ('presentation',)
        assert behavior.shape[0] == 3
        # Each prediction should be a non-empty string
        for val in behavior.values:
            assert isinstance(val, str)
            assert len(val) > 0
