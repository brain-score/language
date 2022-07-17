from datasets import load_dataset
from brainscore_language import datasets

def syntaxgymtestsuite():
    #dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return

testsuitepath="test_suite.json"  # Not sure if this should be here.
datasets['SG-TSE'] = testsuitepath
