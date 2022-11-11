from datasets import load_dataset

from brainscore_language import data_registry


def wikitext2TestFromHuggingface():
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    lines = dataset['text']
    return lines

def miniWikitext2TestFromHuggingface():
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    lines = dataset['text']
    print(len(lines))
    print(lines[100:115])
    # return lines


data_registry['wikitext-2/test'] = wikitext2TestFromHuggingface

if __name__ == '__main__':
    miniWikitext2TestFromHuggingface()