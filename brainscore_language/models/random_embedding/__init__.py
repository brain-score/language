from hashlib import sha256

import numpy as np
from numpy.random import RandomState


def word_to_embedding(word: str, embedding_size: int) -> np.ndarray:
    """
    Create an embedding of size `embedding_size` for the given `word`.
    Embeddings are consistent per word, but different across words (i.e. typically unique).
    """
    # Seed random state to condition on the word.
    # We do not use a global random state to avoid ordering issues when the function is called
    # with word1 first and then word2, or vice-versa.
    word_hash = sha256(word.encode("utf-8"))
    seed = np.frombuffer(word_hash.digest(), dtype='uint32')  # random state seed expects 32-bit unsigned int
    random_state = RandomState(seed)
    embedding = random_state.random(embedding_size)
    return embedding
