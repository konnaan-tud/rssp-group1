"""
observation_pipeline/dynamic_scene.py
---------------------------------------
Manages the dynamic observation sequence for one cooking session.

Each video clip produces one scene sentence (from the VLM or mock).
That sentence is embedded and appended to the session's observation sequence.

The observation sequence grows clip by clip:
  [embed("slices garlic on a cutting board"),
   embed("drops pasta into boiling water"),
   ...]

This sequence is passed to the belief updater after each clip.

Usage:
    from observation_pipeline.dynamic_scene import DynamicScene

    scene = DynamicScene()
    scene.add("slices garlic on a cutting board")
    scene.add("heats olive oil in a skillet")

    # Get current sequence for belief update
    sequence = scene.get_sequence()  # list of np.ndarray

    # Incorporate clarification answer
    scene.add_answer("I am adding pancetta to the skillet")

    print(scene.summary())
"""

from __future__ import annotations

import numpy as np
from observation_pipeline.embedder import Embedder


class DynamicScene:
    """
    Manages the growing observation sequence for one cooking session.

    Each observation (clip or answer) is embedded and stored
    in temporal order. The full sequence is available for DTW matching.

    Parameters
    ----------
    embedder : Embedder instance (created internally if not provided)
    """

    def __init__(self, embedder: Embedder | None = None):
        self._embedder = embedder or Embedder()
        self._sequence: list[np.ndarray] = []
        self._sentences: list[dict] = []  # metadata: sentence + type + index

    # ── Public API ──────────────────────────────────────────────────────

    def add(self, sentence: str) -> np.ndarray:
        """
        Add a new observation scene sentence.
        Embeds it and appends to the sequence.

        Parameters
        ----------
        sentence : scene description from VLM
                   e.g. "slices garlic on a cutting board"

        Returns
        -------
        np.ndarray — the embedded vector (768,)
        """
        sentence = sentence.strip()
        vector = self._embedder.embed(sentence)
        self._sequence.append(vector)
        self._sentences.append({
            "index": len(self._sequence) - 1,
            "type": "observation",
            "sentence": sentence,
        })
        return vector

    def add_answer(self, answer: str) -> np.ndarray:
        """
        Add a clarification answer to the sequence.
        Treated as a special observation at the current position.

        Parameters
        ----------
        answer : natural language answer from human
                 e.g. "I am adding pancetta to the skillet"

        Returns
        -------
        np.ndarray — the embedded vector (768,)
        """
        answer = answer.strip()
        vector = self._embedder.embed(answer)
        self._sequence.append(vector)
        self._sentences.append({
            "index": len(self._sequence) - 1,
            "type": "answer",
            "sentence": answer,
        })
        return vector

    def get_sequence(self) -> list[np.ndarray]:
        """
        Return current observation sequence as ordered list of vectors.
        This is what gets passed to the belief updater.
        """
        return list(self._sequence)

    def get_sentences(self) -> list[dict]:
        """Return metadata for all observations in order."""
        return list(self._sentences)

    def current_length(self) -> int:
        return len(self._sequence)

    def reset(self):
        """Reset for a new session."""
        self._sequence = []
        self._sentences = []

    def summary(self) -> dict:
        return {
            "length": self.current_length(),
            "observations": [
                f"[{s['type']}] {s['sentence']}"
                for s in self._sentences
            ],
        }


if __name__ == "__main__":
    scene = DynamicScene()

    # Simulate a carbonara session
    observations = [
        "pours water into a large pot",
        "cracks eggs into a mixing bowl",
        "grates pecorino into the mixing bowl",
        "dices pancetta on a cutting board",
    ]

    print("=== Dynamic Scene Test ===\n")
    for obs in observations:
        vec = scene.add(obs)
        print(f"  Added: '{obs}'")
        print(f"  Vector shape: {vec.shape}, norm: {np.linalg.norm(vec):.4f}")

    # Add clarification answer
    answer = "I am adding pancetta and eggs to make carbonara"
    vec = scene.add_answer(answer)
    print(f"\n  Answer: '{answer}'")
    print(f"  Vector shape: {vec.shape}")

    print(f"\n=== Summary ===")
    s = scene.summary()
    print(f"  Sequence length: {s['length']}")
    for obs in s["observations"]:
        print(f"  {obs}")

    # Check similarity between related sentences
    seq = scene.get_sequence()
    print(f"\n=== First vs last vector cosine similarity ===")
    v1, v2 = seq[0], seq[-1]
    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    print(f"  {sim:.4f}")