"""Tests for the version-agnostic KV-cache sliding window.

The sliding window is the only thing that forced ``transformers<5``. Each cache
API that transformers has shipped is stood up as a stub here, so all branches are
covered without installing three versions of the library — including the >=5
layout, which cannot be installed alongside the current pin.
"""

import numpy as np
import pytest
import torch

from brainscore_language.model_helpers.huggingface import slice_kv_cache

N_BATCH, N_HEADS, N_POS, HEAD_DIM = 1, 2, 10, 4
KEEP = 3


def _tensor(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(N_BATCH, N_HEADS, N_POS, HEAD_DIM, generator=generator)


class _Layer:
    """transformers >= 5 per-layer cache entry."""

    def __init__(self, keys, values):
        self.keys, self.values = keys, values


class _ModernCache:
    def __init__(self, pairs):
        self.layers = [_Layer(k, v) for k, v in pairs]


class _LegacyRoundTripCache:
    """transformers 4.x DynamicCache surface."""

    def __init__(self, pairs):
        self._pairs = tuple(pairs)

    def to_legacy_cache(self):
        return self._pairs

    @classmethod
    def from_legacy_cache(cls, pairs):
        return cls(pairs)


class _InternalsCache:
    """4.x internals, used only if the legacy round-trip is absent."""

    def __init__(self, pairs):
        self.key_cache = [k for k, _ in pairs]
        self.value_cache = [v for _, v in pairs]


@pytest.fixture
def pairs():
    return [(_tensor(i), _tensor(100 + i)) for i in range(3)]


def _expected(pairs):
    return [(k[:, :, -KEEP:, :], v[:, :, -KEEP:, :]) for k, v in pairs]


class TestSliceKVCache:
    def test_tuple_format(self, pairs):
        out = slice_kv_cache(tuple(pairs), KEEP)
        for (k, v), (ek, ev) in zip(out, _expected(pairs)):
            assert torch.equal(k, ek) and torch.equal(v, ev)

    def test_modern_layers_format(self, pairs):
        out = slice_kv_cache(_ModernCache(pairs), KEEP)
        for layer, (ek, ev) in zip(out.layers, _expected(pairs)):
            assert torch.equal(layer.keys, ek) and torch.equal(layer.values, ev)

    def test_legacy_round_trip_format(self, pairs):
        out = slice_kv_cache(_LegacyRoundTripCache(pairs), KEEP)
        for (k, v), (ek, ev) in zip(out.to_legacy_cache(), _expected(pairs)):
            assert torch.equal(k, ek) and torch.equal(v, ev)

    def test_internals_fallback_format(self, pairs):
        out = slice_kv_cache(_InternalsCache(pairs), KEEP)
        for k, (ek, _) in zip(out.key_cache, _expected(pairs)):
            assert torch.equal(k, ek)
        for v, (_, ev) in zip(out.value_cache, _expected(pairs)):
            assert torch.equal(v, ev)

    def test_every_format_yields_the_same_values(self, pairs):
        """The point of the shim: the API shape must not change the result."""
        from_tuple = [(k, v) for k, v in slice_kv_cache(tuple(pairs), KEEP)]
        from_modern = [(l.keys, l.values)
                       for l in slice_kv_cache(_ModernCache(pairs), KEEP).layers]
        from_legacy = list(
            slice_kv_cache(_LegacyRoundTripCache(pairs), KEEP).to_legacy_cache())
        for a, b, c in zip(from_tuple, from_modern, from_legacy):
            assert torch.equal(a[0], b[0]) and torch.equal(b[0], c[0])
            assert torch.equal(a[1], b[1]) and torch.equal(b[1], c[1])

    def test_keeps_the_newest_positions_not_the_oldest(self, pairs):
        """Guards the direction: ``Cache.crop`` keeps the oldest, we need newest."""
        out = slice_kv_cache(tuple(pairs), KEEP)
        original_key = pairs[0][0]
        assert torch.equal(out[0][0], original_key[:, :, -KEEP:, :])
        assert not torch.equal(out[0][0], original_key[:, :, :KEEP, :])

    def test_unknown_cache_type_raises(self):
        with pytest.raises(TypeError, match='no known transformers cache API'):
            slice_kv_cache(object(), KEEP)


def test_against_the_real_installed_dynamiccache():
    """Cover the actually-installed transformers, not just the stubs above."""
    from transformers import DynamicCache

    pairs = [(_tensor(i), _tensor(100 + i)) for i in range(2)]
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(pairs):
        cache.update(k, v, layer_idx)
    assert cache.get_seq_length() == N_POS

    out = slice_kv_cache(cache, KEEP)
    assert out.get_seq_length() == KEEP

    # Read the result back without assuming which API this version exposes —
    # `to_legacy_cache` is 4.x-only, `.layers` is 5.x.
    if hasattr(out, 'to_legacy_cache'):
        got = list(out.to_legacy_cache())
    else:
        got = [(layer.keys, layer.values) for layer in out.layers]
    for (k, v), (ek, ev) in zip(got, _expected(pairs)):
        assert torch.equal(k, ek) and torch.equal(v, ev)
