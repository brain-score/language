"""SUMA uses a gated Llama tokenizer."""
import pytest


@pytest.fixture(autouse=True)
def authenticate_suma(set_hf_token):
    """Request the shared authentication fixture for SUMA tests."""
