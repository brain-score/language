"""Public model tests must run even when no credentials are available."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize('gated, token, expected', [
    (False, False, '1 passed'),
    (True, False, '1 skipped'),
    (True, True, '1 passed'),
])
def test_auth_is_requested_only_for_gated_models(tmp_path, gated, token, expected):
    source = Path(__file__).parents[2] / 'brainscore_language/models/conftest.py'
    shutil.copyfile(source, tmp_path / 'conftest.py')
    # A public test must never try either authentication service.
    (tmp_path / 'boto3.py').write_text(
        'def client(*args, **kwargs):\n    raise RuntimeError("no credentials")\n')
    (tmp_path / 'huggingface_hub.py').write_text(
        'def login(*args, **kwargs):\n    raise AssertionError("unexpected login")\n')
    marker = '@pytest.mark.usefixtures("set_hf_token")\n' if gated else ''
    (tmp_path / 'test_model.py').write_text(
        'import pytest\n' + marker + 'def test_model():\n    assert True\n')
    environment = dict(os.environ)
    environment.pop('HF_TOKEN', None)
    environment['PYTHONPATH'] = str(tmp_path)
    environment['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    if token:
        environment['HF_TOKEN'] = 'test-token-not-used-for-network'
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', str(tmp_path)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout
