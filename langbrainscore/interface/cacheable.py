

import pickle
import typing
from abc import ABC, abstractclassmethod, abstractmethod
from pathlib import Path
import yaml
import xarray as xr
from langbrainscore.utils.cache import get_cache_root
from langbrainscore.utils.logging import log

T = typing.TypeVar('T')

class _Cacheable(ABC):
    '''
    A class used to define a common interface for Object caching in LangBrainscore
    '''

    # @abstractclassmethod
    # @classmethod
    def _get_xarray_objects(self) -> typing.Iterable[str]:
        '''
        returns all (visible) attributes of self that are instances of xarray
        '''
        keys = []
        for key, ob in vars(self).items():
            if isinstance(ob, xr.DataArray):
                keys += [key]
        return keys

    # @abstractclassmethod
    # def _get_meta_attributes(cls) -> typing.Iterable[str]:
    #     return ()

    def to_cache(self, identifier_string: str, overwrite = True,
                 xarray_serialization_backend='to_zarr', cache_dir = None) -> None:
        '''
        dump this object to cache. this method implementation will serve
        as the default implementation. it is recommended that this be left
        as-is for compatibility with caching across the library.
        '''
        root = Path(cache_dir or get_cache_root()).expanduser().resolve()
        root /= identifier_string
        root /= self.__class__.__name__
        root.mkdir(exist_ok=True, parents=True)
        log(f'caching {self} to {root}')

        with (root / 'xarray_object_names.yml').open('w') as f:
            yaml.dump(self._get_xarray_objects(), f, yaml.SafeDumper)

        kwargs = {}
        if overwrite and 'zarr' in xarray_serialization_backend: 
            kwargs.update({'mode':'w'})
        for ob_name in self._get_xarray_objects():
            ob = getattr(self, ob_name)
            tgt_dir = root / (ob_name + '.xr')
            
            dump_object = getattr(ob.to_dataset(name='data'), xarray_serialization_backend)
            dump_object(tgt_dir, **kwargs)

        with (root / f'{self.__class__.__name__}.pkl').open('wb') as f:
            pickle.dump(self, f)


    # NB comment from Guido: https://github.com/python/typing/issues/58#issuecomment-194569410
    @classmethod
    def from_cache(cls, identifier_string: str, 
                   xarray_deserialization_backend='open_zarr',
                   cache_dir = None) -> T:
        '''
        construct an object from cache. subclasses must start with the
        object returned by a call to this method like so:

            ob = super().from_cache(filename)
            # further implementation, such as initializing
            # member classes based on metadata
            return ob
            
        '''
        root = Path(cache_dir or get_cache_root()).expanduser().resolve()
        root /= identifier_string
        root /= cls.__name__
        root.mkdir(exist_ok=True, parents=True)
        log(f'loading cache for {cls} from {root} {identifier_string}')

        with (root/f'{cls.__name__}.pkl').open('rb') as f:
            ob = pickle.load(f)

        with (root / 'xarray_object_names.yml').open('r') as f:
            xarray_object_names = yaml.load(f, yaml.SafeLoader)

        for attr in xarray_object_names:
            tgt_dir = root / (attr + '.xr')
            load_object = getattr(xr, xarray_deserialization_backend)
            xarray_instance = load_object(tgt_dir)
            setattr(ob, attr, xarray_instance.data)

        return ob
