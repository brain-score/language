'''
utilities related to caching, including creating a directory structure,
managing a disk-backed LRU cache, etc.  
'''

import typing
from pathlib import Path
import os
from dataclasses import dataclass

@dataclass
class CacheDescriptor:
    '''
    A class to conveniently hold various paths within the LBS_CACHE directory structure
    '''
    root: Path
    subdir: Path
    # human_readable_name: Path
    
    def mkdirs(self):
        '''creates directories if they don't already exist'''
        self.subdir.mkdir(parents=True, exist_ok=True)


def pathify(fpth: typing.Union[Path, str, typing.Any]) -> Path:
    '''
    returns a resolved `Path` object after expanding user and shorthands/symlinks
    '''
    return Path(fpth).expanduser().resolve()


def get_cache_directory(prefix: typing.Union[str, Path] = '~/.cache',
                        calling_class = None,
                        # subdirs: typing.List[str] = ['dataset', 'encoder', 'mapping', 'metric', 'brainscore'],
                        # randomize: bool = False
                        ) -> CacheDescriptor:
    '''
    returns the "root" of langbrainscore cache. any instance-specific runs must make sure
    to make their own directory structure within this root and identify themselves uniquely
    so as not to get overwritten by other runs
    '''
    if 'LBS_CACHE' in os.environ: # if environment variable is specified, use that with first priority
        prefix = os.environ['LBS_CACHE']

    prefix = pathify(prefix)
    root = prefix / 'langbrainscore'

    # if randomize:
    #     import randomname
    #     while (root / (human_readable := randomname.generate())).exists():
    #         pass

    CD = CacheDescriptor(root=root, **{'subdir': root / subdir for subdir in [calling_class or 'uncategorized']})
    CD.mkdirs()
    return CD


