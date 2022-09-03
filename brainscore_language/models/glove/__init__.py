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
    weights_file = Path(__file__).parent / f'{name}.txt'
    if not weights_file.is_file():
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
    _logger.info(f"Using weights {weights_file}")
    return weights_file


def glove(name: str, dimensions: int) -> ArtificialSubject:
    """
    Pennington et al., 2014
    www.aclweb.org/anthology/D14-1162
    """
    weights_file = _prepare_weights(name)
    return GensimKeyedVectorsSubject(identifier='glove', weights_file=weights_file, weights_file_no_header=True,
                                     vector_size=dimensions)


model_registry['glove-840b'] = lambda: glove('glove.840B.300d', dimensions=300)
