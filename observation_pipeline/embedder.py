"""
observation_pipeline/embedder.py
---------------------------------
Sentence embedding using all-mpnet-base-v2.

Caches all embeddings to data/embedding_cache.json so the same
sentence is never embedded twice across sessions.

Used by both observation_pipeline and recipe_pipeline.

Usage:
    from observation_pipeline.embedder import Embedder
    embedder = Embedder()
    vector = embedder.embed("slices garlic on a cutting board")
"""

from __future__ import annotations

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-mpnet-base-v2"
DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "embedding_cache.json"
)


class Embedder:
    """
    Embeds sentences into 768-dimensional vectors using all-mpnet-base-v2.
    Caches results to disk — same sentence never embedded twice.

    Parameters
    ----------
    cache_path : path to JSON cache file (default: data/embedding_cache.json)
    """

    def __init__(self, cache_path: str = DEFAULT_CACHE_PATH):
        self.cache_path = os.path.abspath(cache_path)
        self._cache: dict[str, list[float]] = {}

        # Load existing cache
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            print(f"[Embedder] Loaded {len(self._cache)} cached embeddings.")
        else:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        # Load model
        print(f"[Embedder] Loading {MODEL_NAME}...")
        self._model = SentenceTransformer(MODEL_NAME)
        print(f"[Embedder] Ready. Vector size: 768.")

    def embed(self, sentence: str) -> np.ndarray:
        """
        Embed a single sentence.
        Returns cached result if available.

        Parameters
        ----------
        sentence : str

        Returns
        -------
        np.ndarray of shape (768,)
        """
        sentence = sentence.strip()
        if sentence in self._cache:
            return np.array(self._cache[sentence], dtype=np.float32)

        vector = self._model.encode(sentence, convert_to_numpy=True)
        self._cache[sentence] = vector.tolist()
        self._save_cache()
        return vector.astype(np.float32)

    def embed_batch(self, sentences: list[str]) -> list[np.ndarray]:
        """
        Embed multiple sentences efficiently.
        Only calls the model for uncached sentences.

        Parameters
        ----------
        sentences : list of str

        Returns
        -------
        list of np.ndarray, each of shape (768,)
        """
        sentences = [s.strip() for s in sentences]
        uncached = [s for s in sentences if s not in self._cache]

        if uncached:
            vectors = self._model.encode(uncached, convert_to_numpy=True)
            for sentence, vector in zip(uncached, vectors):
                self._cache[sentence] = vector.tolist()
            self._save_cache()

        return [
            np.array(self._cache[s], dtype=np.float32)
            for s in sentences
        ]

    def _save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)

    @property
    def cache_size(self) -> int:
        return len(self._cache)


if __name__ == "__main__":
    embedder = Embedder()

    sentences = [
        "slices garlic on a cutting board",
        "drops pasta into salted boiling water",
        "drains pasta over the sink",
        "cooks pancetta in a skillet",
        "grates pecorino into a bowl",
    ]

    print("\n=== Embedding test ===")
    vectors = embedder.embed_batch(sentences)
    for s, v in zip(sentences, vectors):
        print(f"  '{s[:50]}' → shape {v.shape}, norm {np.linalg.norm(v):.4f}")

    # Check similarity between related sentences
    import itertools
    print("\n=== Pairwise cosine similarities ===")
    for (s1, v1), (s2, v2) in itertools.combinations(zip(sentences, vectors), 2):
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        print(f"  {sim:.4f}  |  '{s1[:35]}' ↔ '{s2[:35]}'")

    print(f"\nCache size: {embedder.cache_size}")