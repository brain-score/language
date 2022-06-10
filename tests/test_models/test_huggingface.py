from brainscore_language.models.huggingface import distilgpt2

from brainscore_language.brainmodel import BrainModel


def test_next_word():
    model = distilgpt2()
    model.start_task(task=BrainModel.Task.next_word)
    text = 'the quick brown fox'
    next_word = model.digest_text(text)
    assert isinstance(next_word, str)
