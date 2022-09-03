import logging
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from gensim.scripts.glove2word2vec import glove2word2vec

from brainscore_language import model_registry, ArtificialSubject
from brainscore_language.model_helpers.embedding import GensimKeyedVectorsSubject

_logger = logging.getLogger(__name__)


def _prepare_weights(name):
    word2vec_weightsfile = Path(__file__).parent / f'{name}.txt.word2vec'
    if not word2vec_weightsfile.is_file():
        # download zip
        zip_filename = f"{name}.zip"
        url = f'https://nlp.stanford.edu/data/{zip_filename}'
        _logger.info(f"Downloading weights zip from {url} to {zip_filename}")
        urlretrieve(url, zip_filename)

        # unzip
        _logger.debug(f"Unzipping {zip_filename}")
        weights_base_directory = Path(__file__).parent
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(weights_base_directory)
        os.remove(zip_filename)

        # convert format
        weights_file = weights_base_directory / f"{name}.txt"
        _logger.debug(f"Converting weights {weights_file} to word2vec format")
        glove2word2vec(weights_file, word2vec_weightsfile)
        os.remove(weights_file)
    _logger.info(f"Using weights {word2vec_weightsfile}")
    return word2vec_weightsfile


def glove(name: str, dimensions: int) -> ArtificialSubject:
    """
    Pennington et al., 2014
    www.aclweb.org/anthology/D14-1162
    """
    weights_file = _prepare_weights(name)
    return GensimKeyedVectorsSubject(identifier='glove', weights_file=weights_file, vector_size=dimensions)


model_registry['glove-840b'] = lambda: glove('glove.840B.300d', dimensions=300)
