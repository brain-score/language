"""Check download routing with a tiny archive, without model inference."""
import importlib.util
from pathlib import Path
import zipfile

import pytest


@pytest.fixture
def glove_module(tmp_path, monkeypatch):
    source = Path(__file__).parents[2] / 'brainscore_language/models/glove/model.py'
    spec = importlib.util.spec_from_file_location('glove_download_under_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, '__file__', str(tmp_path / 'model.py'))
    monkeypatch.chdir(tmp_path)
    return module


def test_download_original_vectors_from_official_host(glove_module, monkeypatch):
    calls = []

    def download(url, filename):
        calls.append(url)
        with zipfile.ZipFile(filename, 'w') as archive:
            archive.writestr('glove.840B.300d.txt', 'test 0.1 0.2\n')

    monkeypatch.setattr(glove_module, 'urlretrieve', download)
    result = glove_module._prepare_weights('glove.840B.300d')
    assert result.read_text() == 'test 0.1 0.2\n'
    assert calls == ['https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip']
    assert not Path('glove.840B.300d.zip').exists()
    assert glove_module._prepare_weights('glove.840B.300d') == result
    assert len(calls) == 1


def test_download_failure_remains_a_failure(glove_module, monkeypatch):
    def download(*args):
        raise OSError('unreachable')

    monkeypatch.setattr(glove_module, 'urlretrieve', download)
    with pytest.raises(OSError, match='unreachable'):
        glove_module._prepare_weights('glove.840B.300d')
