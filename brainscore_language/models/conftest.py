"""
Shared pytest fixtures for all model tests.

Some HuggingFace models (e.g. Gemma) are "gated" -- they require an authenticated
token to download weights. This conftest provides a session-scoped fixture that
automatically fetches a HuggingFace read token from AWS Secrets Manager and logs in
before any model test runs. If the HF_TOKEN environment variable is already set
(e.g. via `huggingface-cli login` or manual export), the AWS lookup is skipped.

Because this file lives in brainscore_language/models/, pytest automatically applies
its fixtures to every test in this directory and all model subdirectories (gemma/,
gpt/, etc.), so individual test files don't need any auth boilerplate.
"""

import json
import os

import boto3
import pytest
from huggingface_hub import login


@pytest.fixture(autouse=True, scope="session")
def set_hf_token():
    """Pull HuggingFace token from AWS Secrets Manager if not already set."""
    if os.environ.get("HF_TOKEN"):
        return
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
