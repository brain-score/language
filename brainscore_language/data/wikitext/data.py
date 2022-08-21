from datasets import load_dataset

from brainscore_language import datasets


def wikitext2TestFromHuggingface():
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    lines = dataset['text']
    return lines


datasets['wikitext-2/test'] = wikitext2TestFromHuggingface
