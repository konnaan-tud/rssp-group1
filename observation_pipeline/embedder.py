from __future__ import annotations

import hashlib
import json
from pathlib import Path


class SentenceEmbedder:
    """
    Lightweight deterministic embedder with on-disk caching.

    This is a placeholder until a real sentence-transformer is wired in.
    """

    def __init__(self, cache_path: str | Path | None = None, dimensions: int = 16):
        self.cache_path = Path(cache_path) if cache_path else None
        self.dimensions = dimensions
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, list[float]]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        with self.cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _persist(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(self._cache, handle, indent=2)

    def embed(self, sentence: str) -> list[float]:
        if sentence in self._cache:
            return self._cache[sentence]

        digest = hashlib.sha256(sentence.encode("utf-8")).digest()
        vector = [round(byte / 255.0, 6) for byte in digest[: self.dimensions]]
        self._cache[sentence] = vector
        self._persist()
        return vector
