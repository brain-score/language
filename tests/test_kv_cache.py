"""Neural extraction preserves upstream full-context FP32 execution."""
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language.model_helpers.preprocessing import prepare_context


@pytest.fixture
def subject(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    vocab = {word: i for i, word in enumerate(
        ['[UNK]', '[EOS]', 'The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'dog'])}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
        eos_token='[EOS]', pad_token='[EOS]', model_max_length=64)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=64,
            n_embd=32, n_layer=2, n_head=2, resid_pdrop=0, embd_pdrop=0,
            attn_pdrop=0)).eval()
    return HuggingfaceSubject('local-regression', {'language_system': 'transformer.h.1'},
                              model=model, tokenizer=tokenizer)


@pytest.mark.parametrize('behavior', [False, True])
def test_neural_matches_independent_full_context_exactly(subject, behavior):
    if behavior:
        subject.start_behavioral_task(ArtificialSubject.Task.next_word)
    subject.start_neural_recording('language_system', ArtificialSubject.RecordingType.fMRI)
    parts = ['The quick', 'brown fox', 'jumps over the dog']
    calls = []
    handle = subject.basemodel.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append((kwargs['input_ids'].shape[1],
            kwargs.get('use_cache'), kwargs.get('past_key_values'))), with_kwargs=True)
    try:
        actual = subject.digest_text(parts)
    finally:
        handle.remove()
    expected = []
    layer = subject._get_layer('transformer.h.1')
    handle = layer.register_forward_hook(lambda module, args, out:
        expected.append((out[0] if isinstance(out, tuple) else out)[:, -1].detach().numpy()))
    try:
        with torch.no_grad():
            for end in range(1, len(parts) + 1):
                tokens = subject.tokenizer(prepare_context(parts[:end]), return_tensors='pt')
                subject.basemodel(**tokens, use_cache=False)
    finally:
        handle.remove()
    np.testing.assert_array_equal(actual['neural'].values, np.concatenate(expected))
    assert calls == [(2, False, None), (4, False, None), (8, False, None)]
    assert (actual['behavior'] is not None) is behavior


def test_behavior_only_retains_incremental_cache(subject):
    subject.start_behavioral_task(ArtificialSubject.Task.next_word)
    calls = []
    handle = subject.basemodel.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append((kwargs['input_ids'].shape[1],
            kwargs.get('use_cache'), kwargs.get('past_key_values') is not None)), with_kwargs=True)
    try:
        result = subject.digest_text(['The quick', 'brown fox', 'jumps over the dog'])
    finally:
        handle.remove()
    assert calls == [(2, True, False), (2, True, True), (4, True, True)]
    assert result['neural'] is None
    assert result['behavior'].shape == (3,)
