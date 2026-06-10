"""
embedder.py
-----------
Wraps sentence-transformers all-mpnet-base-v2 for text embedding.
Caches all embeddings to disk so the same string is never embedded twice.

Usage:
    from embedder import Embedder
    embedder = Embedder()
    vector = embedder.embed("slice cucumber")  # returns np.array (768,)
"""

from __future__ import annotations

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer


CACHE_PATH = "embedding_cache.json"
MODEL_NAME = "all-mpnet-base-v2"


class Embedder:
    """
    Embeds text strings into 768-dimensional vectors using all-mpnet-base-v2.
    Caches results to disk to avoid re-embedding the same string.
    """

    def __init__(self, cache_path: str = CACHE_PATH):
        self.cache_path = cache_path
        self._cache: dict[str, list[float]] = {}

        # Load existing cache from disk
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            print(f"[Embedder] Loaded {len(self._cache)} cached embeddings.")

        # Load model
        print(f"[Embedder] Loading {MODEL_NAME}...")
        self._model = SentenceTransformer(MODEL_NAME)
        print(f"[Embedder] Model ready.")

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a text string. Returns cached result if available.
        """
        text = text.strip()
        if text in self._cache:
            return np.array(self._cache[text], dtype=np.float32)

        vector = self._model.encode(text, convert_to_numpy=True)
        self._cache[text] = vector.tolist()
        self._save_cache()
        return vector.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Embed multiple texts efficiently using batch processing for uncached texts.
        """
        uncached = [t for t in texts if t.strip() not in self._cache]
        if uncached:
            vectors = self._model.encode(uncached, convert_to_numpy=True)
            for text, vector in zip(uncached, vectors):
                self._cache[text.strip()] = vector.tolist()
            self._save_cache()
        return [np.array(self._cache[t.strip()], dtype=np.float32) for t in texts]

    def _save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)

    @property
    def cache_size(self) -> int:
        return len(self._cache)


if __name__ == "__main__":
    embedder = Embedder()

    texts = [
        "slice cucumber",
        "chop onion", 
        "whisk sour cream",
        "capsicum",
        "bell pepper",
    ]

    vectors = embedder.embed_batch(texts)

    print("\n=== Embedding shapes ===")
    for t, v in zip(texts, vectors):
        print(f"  '{t}': {v.shape}")

    cos_sim = np.dot(vectors[3], vectors[4]) / (
        np.linalg.norm(vectors[3]) * np.linalg.norm(vectors[4])
    )
    print(f"\ncosine_sim('capsicum', 'bell pepper') = {cos_sim:.4f}")
    print(f"Cache size: {embedder.cache_size}")