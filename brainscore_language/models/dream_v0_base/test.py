from types import SimpleNamespace

import numpy as np
import pytest
import torch

from brainscore_language import ArtificialSubject, load_model, model_registry
from .subject import BEHAVIOR_CONTEXT_TOKENS, LAYER, DreamV0BaseSubject


@pytest.fixture(autouse=True, scope="session")
def set_hf_token():
    """Override gated-model authentication: Dream is public; unit tests use no weights."""


class _Tokenizer:
    """Deterministic test tokenizer; no checkpoint or network access."""

    words = {"one": 3, "two": 4, "three": 5, "four": 6}

    def __init__(self):
        self.calls = []

    def __call__(self, text, *, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        ids = [self.words[word] for word in text.split()]
        return {"input_ids": ([2] if add_special_tokens else []) + ids}


class _Model:
    config = SimpleNamespace(mask_token_id=31, max_position_embeddings=4096)

    def __init__(self):
        self.calls = []

    @staticmethod
    def raw_logits(input_ids):
        # Each position favors its input token's successor. The preceding
        # observed and mask positions thus have distinct target distributions.
        centers = (input_ids + 1) % 32
        vocabulary = torch.arange(32, dtype=torch.float32)
        return -torch.abs(vocabulary - centers[..., None]) * 0.25

    def __call__(self, *, input_ids, use_cache, num_logits_to_keep=0,
                 output_hidden_states=False, return_dict=False):
        self.calls.append({"input_ids": input_ids.clone(), "use_cache": use_cache,
                           "num_logits_to_keep": num_logits_to_keep,
                           "output_hidden_states": output_hidden_states})
        # Pinned Dream slices hidden states before its output head. Emulate
        # that behavior so requesting only the last position cannot pass.
        kept = input_ids[:, -num_logits_to_keep:] if num_logits_to_keep else input_ids
        logits = self.raw_logits(kept)
        hidden_states = None
        if output_hidden_states:
            positions = torch.arange(input_ids.shape[1], dtype=torch.float32)
            hidden = torch.stack((input_ids[0].float(), positions), dim=-1)[None]
            hidden_states = tuple(hidden for _ in range(LAYER + 1))
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


@pytest.fixture
def fake_subject():
    subject = DreamV0BaseSubject.__new__(DreamV0BaseSubject)
    subject.tokenizer = _Tokenizer()
    subject.model = _Model()
    subject.input_device = torch.device("cpu")
    subject._recording = None
    subject._behavioral_task = None
    return subject


def _native_mask_bits(query_ids, target_id):
    """Oracle uses the pinned native shift, independently of plugin indexing."""
    raw = _Model.raw_logits(torch.tensor([query_ids]))
    shifted = torch.cat([raw[:, :1], raw[:, :-1]], dim=1)
    return -torch.log_softmax(shifted[0, -1], dim=-1)[target_id].item() / np.log(2)


def test_behavior_matches_native_shift_not_mask_position(fake_subject):
    bits, context = fake_subject._part_surprisal(["one", "two"])
    assert context == "one two"
    assert bits == pytest.approx(_native_mask_bits([2, 3, 31], 4))
    wrong_mask_bits = -torch.log_softmax(
        _Model.raw_logits(torch.tensor([[2, 3, 31]]))[0, -1], dim=-1
    )[4].item() / np.log(2)
    assert bits != pytest.approx(wrong_mask_bits)
    assert fake_subject.model.calls[0]["num_logits_to_keep"] == 2


def test_mask_only_query_matches_native_first_position(fake_subject):
    bits, _ = fake_subject._part_surprisal(["one"])
    assert bits == pytest.approx(_native_mask_bits([31], 3))
    assert fake_subject.model.calls[0]["input_ids"].tolist() == [[31]]
    assert fake_subject.tokenizer.calls == [("one", False)]


def test_multitoken_chain_hides_current_and_future_targets(fake_subject):
    bits, _ = fake_subject._part_surprisal(["one", "two three four"])
    expected_queries = [[2, 3, 31], [2, 3, 4, 31], [2, 3, 4, 5, 31]]
    assert [call["input_ids"].tolist()[0] for call in fake_subject.model.calls] == expected_queries
    assert bits == pytest.approx(sum(
        _native_mask_bits(query, target)
        for query, target in zip(expected_queries, [4, 5, 6])
    ))
    assert fake_subject.tokenizer.calls == [("one", True), (" two three four", False)]
    assert all(call["num_logits_to_keep"] == 2 and not call["use_cache"]
               for call in fake_subject.model.calls)


def test_current_region_does_not_change_prefix_tokenization(fake_subject):
    fake_subject._part_surprisal(["one", "two"])
    first_query = fake_subject.model.calls[0]["input_ids"].tolist()
    fake_subject.model.calls.clear()
    fake_subject.tokenizer.calls.clear()
    fake_subject._part_surprisal(["one", "three four"])
    assert fake_subject.model.calls[0]["input_ids"].tolist() == first_query
    assert fake_subject.tokenizer.calls[0] == ("one", True)


def test_empty_region_zero_and_omitted_from_later_context(fake_subject):
    bits, context = fake_subject._part_surprisal(["one", " "])
    assert bits == 0
    assert context == "one"
    assert fake_subject.model.calls == []
    after_empty, context = fake_subject._part_surprisal(["one", " ", "two"])
    assert context == "one two"
    assert after_empty == pytest.approx(_native_mask_bits([2, 3, 31], 4))
    fake_subject.model.calls.clear()
    assert fake_subject._part_surprisal([""]) == (0.0, "")
    assert fake_subject.model.calls == []


@pytest.mark.parametrize("prefix_length", [4094, 4095, 4096])
def test_observed_context_cap_preserves_native_readout(fake_subject, prefix_length):
    bits, _ = fake_subject._part_surprisal([" ".join(["one"] * prefix_length), "two"])
    expected_observed = ([2] + [3] * prefix_length)[-BEHAVIOR_CONTEXT_TOKENS:]
    query = fake_subject.model.calls[0]["input_ids"]
    assert query.tolist() == [expected_observed + [31]]
    assert query.shape[1] == BEHAVIOR_CONTEXT_TOKENS + 1
    assert bits == pytest.approx(_native_mask_bits(expected_observed + [31], 4))


def test_clean_neural_readout_retains_current_region_mean(fake_subject):
    value, context = fake_subject._part_neural(["one", "two three"])
    assert context == "one two three"
    np.testing.assert_allclose(value, [4.5, 2.5])
    call = fake_subject.model.calls[0]
    assert call["input_ids"].tolist() == [[2, 3, 4, 5]]
    assert call["output_hidden_states"]
    assert call["num_logits_to_keep"] == 0


def test_behavior_digest_returns_region_bits_including_empty(fake_subject):
    fake_subject.start_behavioral_task(ArtificialSubject.Task.reading_times)
    output = fake_subject.digest_text(["one", "", "two three"])
    assert output["neural"] is None
    np.testing.assert_allclose(output["behavior"].values, [
        _native_mask_bits([31], 3), 0,
        _native_mask_bits([2, 3, 31], 4) + _native_mask_bits([2, 3, 4, 31], 5),
    ], rtol=1e-6)
    assert output["behavior"].coords["context"].values.tolist() == ["one", "one", "one two three"]


def test_registration():
    __import__("brainscore_language.models.dream_v0_base")
    assert "dream-v0-base-7b" in model_registry


def test_load_registered_model_without_checkpoint(fake_subject, monkeypatch):
    from brainscore_language.models import dream_v0_base

    # Exercise the real registry and load_model entry point while replacing
    # only the heavyweight constructor in the registered factory's globals.
    monkeypatch.setattr(dream_v0_base, "DreamV0BaseSubject", lambda: fake_subject)
    model = load_model("dream-v0-base-7b")
    assert model is fake_subject
    assert model.identifier == "dream-v0-base-7b"


@pytest.mark.memory_intense
def test_load_model():
    model = load_model("dream-v0-base-7b")
    assert model.identifier == "dream-v0-base-7b"
