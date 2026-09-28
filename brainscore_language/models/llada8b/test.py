import pytest
import torch

from brainscore_language import ArtificialSubject, load_model, model_registry


def test_registration():
    __import__("brainscore_language.models.llada8b")
    assert "llada-8b-base" in model_registry


@pytest.mark.memory_intense
def test_load_model():
    model = load_model("llada-8b-base")
    # Brain-Score's loader sets ``model.identifier`` to the registered string.
    assert model.identifier == "llada-8b-base"


def test_reading_times_hide_current_part_and_allow_blank_regions():
    from brainscore_language.models.llada8b.subject import LLaDA8BSubject

    class Tokenizer:
        def __call__(self, text, *, add_special_tokens):
            del add_special_tokens
            return {"input_ids": [{"a": 1, "b": 2, " ": 3}[c] for c in text]}

    class Model:
        config = type("Config", (), {"mask_token_id": 9, "max_sequence_length": 16})()

        def __init__(self):
            self.seen = []
            self.model = self

        def __call__(self, *, input_ids, last_logits_only):
            assert last_logits_only is True
            self.seen.append(input_ids.tolist()[0])
            logits = torch.zeros((1, input_ids.shape[1], 10))
            return type("Output", (), {"logits": logits})()

    subject = LLaDA8BSubject.__new__(LLaDA8BSubject)
    subject.tokenizer, subject.model = Tokenizer(), Model()
    subject.input_device = torch.device("cpu")
    subject._recording = None
    subject._behavioral_task = None
    subject.start_behavioral_task(ArtificialSubject.Task.reading_times)
    output = subject.digest_text(["a", "b", ""])
    assert output["behavior"].shape == (3,)
    assert output["behavior"].values[2] == 0
    assert output["neural"] is None
    # The toy tokenizer splits the leading space and the target letter.
    assert subject.model.seen == [[9], [1, 9], [1, 3, 9]]
    subject.model.seen.clear()
    subject.digest_text(["a", "", "b"])
    with_empty = list(subject.model.seen)
    subject.model.seen.clear()
    subject.digest_text(["a", "b"])
    assert subject.model.seen == with_empty
