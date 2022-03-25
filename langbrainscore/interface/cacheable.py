

from abc import ABC, abstractmethod, abstractclassmethod
import typing
from pathlib import Path
from diskcache import Cache


T = typing.TypeVar('T', 'Cacheable')

class Cacheable(ABC):
    '''
    A class used to define a common interface for Object caching in LangBrainscore
    '''

    @abstractclassmethod
    def _get_netcdf_cacheable_objects(cls) -> typing.Iterable[str]:
        return ()

    @abstractclassmethod
    def _get_meta_attributes(cls) -> typing.Iterable[str]:
        return ()

    def to_cache(self, filename) -> None:
        '''
        dump this object to cache. this method implementation will serve
        as the default implementation. it is recommended that this be left
        as-is for compatibility with caching across the library.
        '''
        NotImplemented

    def from_cache(cls: T, filename) -> T:
        '''
        construct an object from cache. subclasses must start with the
        object returned by a call to this method like so:

            ob = super().from_cache(filename)
            # further implementation, such as initializing
            # member classes based on metadata
            return ob
            
        '''
        return NotImplemented

        C = Cache()
        ob = cls()
        for attr in cls._get_meta_attributes():
            thing = None # retrieve from cache
            setattr(ob, attr, thing)

        return ob