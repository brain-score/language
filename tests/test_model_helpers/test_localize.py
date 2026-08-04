import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from brainscore_language.model_helpers import localize


def _fake_representations(layer_names, hidden_dim, n_samples=8, seed=0):
    rng = np.random.RandomState(seed)
    sentences = {name: rng.randn(n_samples, hidden_dim) + 1 for name in layer_names}
    non_words = {name: rng.randn(n_samples, hidden_dim) for name in layer_names}
    return {"sentences": sentences, "non-words": non_words}


def _run_localize(monkeypatch, cache_dir, top_k, layer_names, hidden_dim,
                  model_id="Qwen/Qwen3-4B"):
    def fake_extract(*args, **kwargs):
        return _fake_representations(layer_names, hidden_dim)

    monkeypatch.setattr(localize, "BRAINIO_CACHE", str(cache_dir))
    monkeypatch.setattr(localize, "extract_representations", fake_extract)
    return localize.localize_fedorenko2010(
        model_id=model_id, model=None, top_k=top_k, tokenizer=None,
        hidden_dim=hidden_dim, layer_names=layer_names, batch_size=4, device="cpu")


def test_top_k_ratio_selects_fraction_of_units(monkeypatch, tmp_path):
    mask = _run_localize(monkeypatch, tmp_path, 0.1, ["layer.0", "layer.1"], 50)
    assert mask.shape == (2, 50)
    assert mask.sum() == 10


def test_top_k_int_is_absolute_count(monkeypatch, tmp_path):
    mask = _run_localize(monkeypatch, tmp_path, 7, ["layer.0", "layer.1"], 50)
    assert mask.sum() == 7


def test_mask_cached_to_slash_replaced_path(monkeypatch, tmp_path):
    mask = _run_localize(monkeypatch, tmp_path, 7, ["layer.0", "layer.1"], 50)
    assert (tmp_path / "Qwen_Qwen3-4B_language_mask.npy").exists()

    def _fail(*args, **kwargs):
        raise AssertionError("should load from cache")

    monkeypatch.setattr(localize, "extract_representations", _fail)
    reloaded = localize.localize_fedorenko2010(
        model_id="Qwen/Qwen3-4B", model=None, top_k=7, tokenizer=None,
        hidden_dim=50, layer_names=["layer.0", "layer.1"], batch_size=4, device="cpu")
    np.testing.assert_array_equal(reloaded, mask)


def test_localize_runs_on_bf16_model(monkeypatch, tmp_path):
    # bf16 activations must be cast before numpy conversion; a completed run proves the fix
    monkeypatch.setattr(localize, "BRAINIO_CACHE", str(tmp_path))
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    layer_names = ["transformer.h.0", "transformer.h.1"]
    hidden_dim = model.config.n_embd
    mask = localize.localize_fedorenko2010(
        model_id="distilgpt2", model=model, top_k=0.1, tokenizer=tokenizer,
        hidden_dim=hidden_dim, layer_names=layer_names, batch_size=8, device="cpu")
    assert mask.shape == (len(layer_names), hidden_dim)
    assert set(np.unique(mask)) <= {0, 1}
    assert mask.sum() == int(0.1 * len(layer_names) * hidden_dim)
