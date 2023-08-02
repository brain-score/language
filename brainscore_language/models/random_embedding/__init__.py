import numpy as np
from hashlib import sha256
from numpy.random import RandomState

from brainscore_language import model_registry
from brainscore_language.model_helpers.embedding import EmbeddingSubject


class WordToEmbedding:
    """
    Create an embedding of size `embedding_size` for a given `word`.
    Embeddings are consistent per word, but different across words (i.e. typically unique).

    Adapted from Schrimpf et al. 2021 https://www.pnas.org/content/118/45/e2105646118,
    https://github.com/mschrimpf/neural-nlp/blob/cedac1f868c8081ce6754ef0c13895ce8bc32efc/neural_nlp/models/implementations.py#L124
    """

    def __init__(self, embedding_size: int):
        self.embedding_size = embedding_size

    def __getitem__(self, word) -> np.ndarray:
        # Seed random state to condition on the word.
        # We do not use a global random state to avoid ordering issues when the function is called
        # with word1 first and then word2, or vice-versa.
        word_hash = sha256(word.encode("utf-8"))
        seed = np.frombuffer(word_hash.digest(), dtype='uint32')  # random state seed expects 32-bit unsigned int
        random_state = RandomState(seed)
        embedding = random_state.random(self.embedding_size)
        return embedding


model_registry['randomembedding-1600'] = lambda: EmbeddingSubject(identifier='randomembedding-1600',
                                                                  lookup=WordToEmbedding(1600))
model_registry['randomembedding-100'] = lambda: EmbeddingSubject(identifier='randomembedding-100',
                                                                 lookup=WordToEmbedding(100))
model_registry['randomembedding-1234'] = lambda: EmbeddingSubject(identifier='randomembedding-1234',
                                                                 lookup=WordToEmbedding(1234))
