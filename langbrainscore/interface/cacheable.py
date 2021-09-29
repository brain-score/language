import pickle
import typing
from abc import ABC, abstractclassmethod, abstractmethod
from numbers import Number
from pathlib import Path

import xarray as xr
import numpy as np
import yaml
from langbrainscore.utils.cache import get_cache_directory, pathify
from langbrainscore.utils.logging import log

# from langbrainscore.interface.dryrunnable import _DryRunnable

T = typing.TypeVar("T")


@typing.runtime_checkable
class _Cacheable(typing.Protocol):
    """
    A class used to define a common interface for Object caching in LangBrainscore
    """

    def __eq__(o1: "_Cacheable", o2: "_Cacheable") -> bool:
        def checkattr(key) -> bool:
            """helper function to check if an attribute is the same between two objects
            and handles AttributeError while at it. if the attributes differ (or does
            not exist on one or the other object), returns False.
            """
            try:
                if getattr(o1, key) != getattr(o2, key):
                    return False
            except AttributeError:
                return False
            return True

        for key, ob in vars(o1).items():
            if isinstance(ob, (str, Number, bool, _Cacheable, tuple, type(None))):
                if not checkattr(key):
                    log(f"{o1} and {o2} differ on {key}", cmap="ERR")
                    return False
            elif isinstance(ob, xr.DataArray):
                x1 = getattr(o1, key)
                x2 = getattr(o2, key)
                if (not np.allclose(x1.data, x2.data, equal_nan=True, atol=1e-4)) or (
                    x1.attrs != x2.attrs
                ):
                    log(f"{o1} and {o2} differ on {key}", cmap="ERR")
                    return False
        else:
            return True

    # @abstractclassmethod
    # @classmethod
    def _get_xarray_objects(self) -> typing.Iterable[str]:
        """
        returns the *names* of all attributes of self that are instances of xarray
        NOTE: this method should be implemented by any subclass irrespective of instance
            state so that in the future we can support loading from cache without having
            to re-run the pipeline (and thereby assign attributes as appropriate)
        by default, just goes over all the objects and returns their names if they are instances
        of `xr.DataArray`
        """
        keys = []
        for key, ob in vars(self).items():
            if isinstance(ob, xr.DataArray):
                keys += [key]
        return keys

    @property
    def params(self) -> dict:
        """ """
        params = {}
        for key in sorted(vars(self)):
            ob = getattr(self, key)
            if isinstance(ob, (str, Number, bool, _Cacheable, tuple, dict, type(None))):
                # if isinstance(ob, (str, Number, bool, _Cacheable, tuple)):
                if isinstance(ob, _Cacheable):
                    params[key] = ob.identifier_string
                elif isinstance(ob, dict):
                    for k in ob:
                        params[f"{key}_{k}"] = ob[k]
                    pass  # TODO!!
                else:
                    params[key] = ob
        return params

    def __repr__(self) -> str:
        """
        default, broad implementation to support our use case.
        constructs a string by concatenating all str, numeric, boolean
        attributes of self, as well as all the representations of Cacheable
        instances that are attributes of self.
        """
        left = "("
        right = ")"
        sep = "?"
        rep = f"{left}{self.__class__.__name__}"
        params = self.params
        for key in sorted([*params.keys()]):
            val = params[key]
            rep += f"{sep}{key}={val}"
        return rep + f"{right}"

    @property
    def identifier_string(self):
        """
        This property aims to return an unambiguous representation of this _Cacheable
        instance, complete with all scalar parameters used to initialize it, and any
        _Cacheable instances that are attributes of this object.

        Unless overridden, makes a call to `repr`
        """
        return repr(self)

    def to_cache(
        self,
        identifier_string: str = None,
        overwrite=True,
        cache_dir=None,
        xarray_serialization_backend="to_zarr",
    ) -> Path:
        """
        dump this object to cache. this method implementation will serve
        as the default implementation. it is recommended that this be left
        as-is for compatibility with caching across the library.

        Args:
            identifier_string (str): a unique identifier string to identify this cache
                instance by (optional; by default, the .identifier_string property is used)
            overwrite (bool): whether to overwrite existing cache by the same identity,
                if it exists. if False, an exce
        """
        if cache_dir:
            cache = get_cache_directory(
                cache_dir, calling_class=self.__class__.__name__
            )
        else:
            cache = get_cache_directory(calling_class=self.__class__.__name__)

        root, subdir = cache.root, cache.subdir
        # now we use "subdir" to be our working directory to dump this cache object
        subdir /= identifier_string or self.identifier_string
        subdir.mkdir(parents=True, exist_ok=overwrite)
        log(f"caching {self} to {subdir}")

        with (subdir / "xarray_object_names.yml").open("w") as f:
            yaml.dump(self._get_xarray_objects(), f, yaml.SafeDumper)
        with (subdir / "id.txt").open("w") as f:
            f.write(self.identifier_string)

        kwargs = {}
        if overwrite and "zarr" in xarray_serialization_backend:
            kwargs.update({"mode": "w"})
        for ob_name in self._get_xarray_objects():
            ob = getattr(self, ob_name)
            tgt_dir = subdir / (ob_name + ".xr")
            dump_object_fn = getattr(
                ob.to_dataset(name="data"), xarray_serialization_backend
            )
            dump_object_fn(tgt_dir, **kwargs)

        cacheable_ptrs = {}
        meta_attributes = {}
        for key, ob in vars(self).items():
            if isinstance(ob, _Cacheable):
                dest = ob.to_cache(
                    identifier_string=identifier_string,
                    overwrite=overwrite,
                    xarray_serialization_backend=xarray_serialization_backend,
                    cache_dir=cache_dir,
                )
                cacheable_ptrs[key] = str(dest)
            elif isinstance(ob, (str, Number, bool, _Cacheable, type(None))):
                meta_attributes[key] = ob
        with (subdir / "meta_attributes.yml").open("w") as f:
            yaml.dump(meta_attributes, f, yaml.SafeDumper)
        with (subdir / "cacheable_object_pointers.yml").open("w") as f:
            yaml.dump(cacheable_ptrs, f, yaml.SafeDumper)

        return subdir

    def load_cache(
        self,
        identifier_string: str = None,
        overwrite: bool = True,
        xarray_deserialization_backend="open_zarr",
        cache_dir=None,
    ) -> Path:
        """load attribute objects from cache onto the existing initialized object (self)"""

        if cache_dir:
            cache = get_cache_directory(
                cache_dir, calling_class=self.__class__.__name__
            )
        else:
            cache = get_cache_directory(calling_class=self.__class__.__name__)

        root, subdir = cache.root, cache.subdir
        # now we use "subdir" as our working directory to dump this cache object
        subdir /= identifier_string or self.identifier_string
        log(f"attempt loading attributes of {self} from {subdir.parent}")

        with (subdir / "xarray_object_names.yml").open("r") as f:
            self_xarray_objects = yaml.load(f, yaml.SafeLoader)

        with (subdir / "id.txt").open("r") as f:
            if (identifier_string or self.identifier_string) != (
                cached_identifier_str := f.read()
            ):
                if not overwrite:
                    raise ValueError(
                        f"mismatch in identifier string of self ({self.identifier_string}) and "
                        f"cached object ({cached_identifier_str}); overwriting is disabled."
                    )
                else:
                    log(
                        f"mismatch in identifier string of self ({self.identifier_string}) and "
                        f"cached object ({cached_identifier_str}); overwriting anyway."
                    )

        kwargs = {}
        for ob_name in self_xarray_objects:
            tgt_dir = subdir / (ob_name + ".xr")
            load_object_fn = getattr(xr, xarray_deserialization_backend)
            ob = load_object_fn(tgt_dir, **kwargs)
            setattr(self, ob_name, ob.data)

        with (subdir / "cacheable_object_pointers.yml").open("r") as f:
            cacheable_ptrs: dict = yaml.load(f, yaml.SafeLoader)

        # calls `load_cache` on all attributes that are also `_Cacheable` instances
        # and thus implement the `load_cache` method
        for key, ptr in cacheable_ptrs.items():
            try:
                ob = getattr(self, key)
                ob.load_cache(
                    identifier_string=identifier_string,
                    overwrite=overwrite,
                    xarray_deserialization_backend=xarray_deserialization_backend,
                    cache_dir=cache_dir,
                )
            except AttributeError:
                log(
                    f"`load_cache` currently only supports loading xarray objects or initialized `_Cacheable` objects"
                )

        with (subdir / "meta_attributes.yml").open("r") as f:
            meta_attributes: dict = yaml.load(f, yaml.SafeLoader)
        for key, ob in meta_attributes.items():
            setattr(self, key, ob)

    # NB comment from Guido: https://github.com/python/typing/issues/58#issuecomment-194569410
    @classmethod
    def from_cache(
        cls: typing.Callable[..., T],
        identifier_string: str,
        xarray_deserialization_backend="open_zarr",
        cache_dir=None,
    ) -> T:
        """
        construct an object from cache. subclasses must start with the
        object returned by a call to this method like so:

            ob = super().from_cache(filename)
            # further implementation, such as initializing
            # member classes based on metadata
            return ob

        """

        Duck = type(cls.__name__, (cls,), {"__init__": (lambda _: None)})
        duck = Duck()
        duck.load_cache(
            identifier_string,
            overwrite=True,
            xarray_deserialization_backend=xarray_deserialization_backend,
            cache_dir=cache_dir,
        )
        return duck
