"""Authentication for model tests that explicitly request ``set_hf_token``.

Public and local models do not need credentials. Gated-model test modules can
use ``pytestmark = pytest.mark.usefixtures("set_hf_token")``. An existing
HF_TOKEN takes precedence over the CI secret lookup.
"""

import json
import os

import pytest


@pytest.fixture(scope="session")
def set_hf_token():
    """Pull HuggingFace token from AWS Secrets Manager if not already set."""
    if os.environ.get("HF_TOKEN"):
        return
    import boto3
    from huggingface_hub import login

    try:
        client = boto3.client("secretsmanager", region_name="us-east-2")
        resp = client.get_secret_value(SecretId="hugging_face_read_token")
        secret = resp["SecretString"]
        # Handle both plain string ("hf_...") and JSON ({"key": "hf_..."}) formats
        try:
            parsed = json.loads(secret)
            if isinstance(parsed, dict):
                token = next(iter(parsed.values()))
            else:
                token = str(parsed)
        except (json.JSONDecodeError, StopIteration):
            token = secret
        token = token.strip()
        os.environ["HF_TOKEN"] = token
        login(token=token)
    except Exception as e:
        pytest.skip(f"HF_TOKEN not set and unable to fetch from AWS Secrets Manager: {e}")
