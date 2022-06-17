# from brainscore_language.models.huggingface import distilgpt2
# from brainscore_language.brainmodel import BrainModel

from brainscore_language.models.huggingface import HuggingfaceModel

def test_next_word():
    # model = distilgpt2()
    # model.start_task(task=BrainModel.Task.next_word)
    # text = 'the quick brown fox'
    # next_word = model.digest_text(text)
    # assert isinstance(next_word, str)

    # from brainscore_language.brainmodel import BrainModel
    temp = HuggingfaceModel(model_id='distilgpt2')
    print('Running', temp.identifier(), 'for next word prediction' )
    temp.start_task(BrainModel.Task.next_word)
    text = 'the quick brown fox'
    next_word = temp.digest_text(text)
    assert next_word == 'es'
    print('next_word:', next_word)
