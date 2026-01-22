"""
Test embedding model for workflow testing.

This is a minimal, fast-running model designed for testing the plugin submission workflow.
It uses a simple hash-based embedding approach similar to random_embedding but optimized for speed.
"""
import numpy as np
from hashlib import md5

from brainscore_language import model_registry
from brainscore_language.model_helpers.embedding import EmbeddingSubject


class SimpleTestEmbedding:
    """
    Simple test embedding that creates consistent embeddings based on word hashes.
    Optimized for speed with small embedding dimensions.
    """

    def __init__(self, embedding_size: int = 50):
        self.embedding_size = embedding_size

    def __getitem__(self, word: str) -> np.ndarray:
        # Use MD5 hash for fast, consistent embeddings
        word_hash = md5(word.encode("utf-8")).digest()
        # Convert hash bytes to float array
        hash_ints = np.frombuffer(word_hash, dtype=np.uint8)
        # Normalize to [-1, 1] range for better numerical properties
        embedding = (hash_ints[:self.embedding_size].astype(np.float32) / 127.5) - 1.0
        # Pad if needed (shouldn't happen with MD5, but just in case)
        if len(embedding) < self.embedding_size:
            padding = np.zeros(self.embedding_size - len(embedding), dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
        return embedding[:self.embedding_size]


# Register the test model with a small embedding size for fast execution
model_registry['test-embedding'] = lambda: EmbeddingSubject(
    identifier='test-embedding',
    lookup=SimpleTestEmbedding(embedding_size=50)
)
