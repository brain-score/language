import logging
from pathlib import Path

from gensim.scripts.glove2word2vec import glove2word2vec

from brainscore_language import model_registry, ArtificialSubject
from brainscore_language.model_helpers.embedding import GensimKeyedVectorsSubject

_logger = logging.getLogger(__name__)


def glove() -> ArtificialSubject:
    """
    Pennington et al., 2014
    http://www.aclweb.org/anthology/D14-1162
    """
    weights_file = Path(__file__).parent / 'glove.840B.300d.txt'
    word2vec_weightsfile = weights_file.parent / (weights_file.name + '.word2vec')
    if not word2vec_weightsfile.is_file():
        _logger.info(f"Converting weights {weights_file} to word2vec format")
        glove2word2vec(weights_file, word2vec_weightsfile)
    return GensimKeyedVectorsSubject(identifier='glove', weights_file=word2vec_weightsfile, vector_size=300)


model_registry['glove'] = glove
