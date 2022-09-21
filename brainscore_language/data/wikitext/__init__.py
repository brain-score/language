from datasets import load_dataset

from brainscore_language import data_registry


def wikitext2TestFromHuggingface():
    dataset = load_data('wikitext', 'wikitext-2-raw-v1', split='test')
    lines = dataset['text']
    return lines


data_registry['wikitext-2/test'] = wikitext2TestFromHuggingface
