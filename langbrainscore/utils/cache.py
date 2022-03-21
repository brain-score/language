'''
utilities related to caching, including creating a directory structure,
managing a disk-backed LRU cache, etc.  
'''

import typing
from pathlib import Path

def get_cache_root(prefix: typing.Union[str, Path] = '~/.cache') -> Path:
    '''

    '''
    prefix = Path(prefix).expanduser().resolve()
    root = prefix / 'langbrainscore'

    act = root / 'encoder_activations'
    data = root / 'datasets' 
    results = root / 'results'

    root.mkdir(parents=True, exist_ok=True)
    return prefix