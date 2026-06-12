"""
probability/belief_updater_v3.py
---------------------------------
Hybrid similarity belief updater over candidate recipes.

After each observation clip or clarification answer:
  1. The new sentence is embedded and appended to the observation sequence
  2. Hybrid similarity (F1 + order consistency) computed against each recipe
  3. Contrastive softmax gives probability distribution
  4. Entropy measured — IG_Q captured for clarification answers

Key design decisions:
  - No Bayesian carry-forward: distribution recomputed from scratch each step
  - Contrastive similarity: raw similarities minus their mean before softmax.
    Generic scenes that match all recipes equally produce zero contrast and
    do not move the distribution. Distinctive scenes produce meaningful
    contrast and drive strong updates.
  - Fixed temperature: principled and easy to justify

Usage:
    from probability.belief_updater_v3 import BeliefUpdaterV3

    updater = BeliefUpdaterV3()
    updater.update("slices garlic on a cutting board")
    updater.update("heats olive oil in a skillet")
    updater.incorporate_answer("I am adding pancetta to the pan")
    print(updater.summary())
"""

from __future__ import annotations

import json
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observation_pipeline.embedder import Embedder
from observation_pipeline.dynamic_scene import DynamicScene
from probability.similarity import hybrid_similarity

RECIPE_SEQUENCES_PATH = os.path.join("data", "recipe_sequences.json")


class BeliefUpdaterV3:
    """
    Contrastive hybrid similarity belief updater over candidate recipes.

    Parameters
    ----------
    recipe_sequences_path : path to data/recipe_sequences.json
    temperature           : softmax temperature (lower = sharper distribution)
    """

    def __init__(
        self,
        recipe_sequences_path: str = RECIPE_SEQUENCES_PATH,
        temperature: float = 0.05,
    ):
        self.temperature = temperature

        # Load recipe sequences
        with open(recipe_sequences_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Convert to numpy arrays
        self.recipe_sequences: dict[str, list[np.ndarray]] = {
            dish: [np.array(v, dtype=np.float32) for v in vectors]
            for dish, vectors in raw.items()
        }

        self.recipe_names = list(self.recipe_sequences.keys())
        self.N = len(self.recipe_names)

        # Shared embedder
        self._embedder = Embedder()

        # Dynamic observation scene
        self._scene = DynamicScene(embedder=self._embedder)

        # Current belief distribution — uniform at start
        self.belief: dict[str, float] = {
            name: 1.0 / self.N for name in self.recipe_names
        }

        # History of all belief states for analysis
        self.history: list[dict[str, float]] = [dict(self.belief)]

        # IG_Q tracking
        self._entropy_before_answer: float | None = None
        self._ig_q_log: list[dict] = []
        self._last_similarities: dict[str, float] = {}
        self._last_contrastive: dict[str, float] = {}

        print(f"[BeliefUpdaterV3] Loaded {self.N} recipes.")
        print(f"[BeliefUpdaterV3] Temperature: {self.temperature}")

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, sentence: str) -> dict[str, float]:
        """
        Process one observation clip.

        Embeds the scene sentence, appends to observation sequence,
        recomputes belief distribution via contrastive similarity.

        Parameters
        ----------
        sentence : scene description e.g. "slices garlic on a cutting board"

        Returns updated belief distribution.
        """
        self._scene.add(sentence)
        return self._recompute()

    def incorporate_answer(self, answer: str) -> dict[str, float]:
        """
        Incorporate a clarification answer.

        Embeds the answer, appends to observation sequence,
        recomputes belief. Records IG_Q.

        Parameters
        ----------
        answer : natural language answer e.g. "I am adding pancetta"

        Returns updated belief distribution.
        """
        self._entropy_before_answer = self.entropy()

        self._scene.add_answer(answer)
        result = self._recompute()

        entropy_after = self.entropy()
        ig = self._entropy_before_answer - entropy_after

        self._ig_q_log.append({
            "answer": answer,
            "entropy_before": round(self._entropy_before_answer, 4),
            "entropy_after": round(entropy_after, 4),
            "ig_q": round(ig, 4),
        })

        return result

    def entropy(self) -> float:
        """Shannon entropy of current belief in bits."""
        return -sum(
            p * math.log2(p)
            for p in self.belief.values()
            if p > 0
        )

    def max_entropy(self) -> float:
        """Maximum possible entropy for N recipes."""
        return math.log2(self.N)

    def top_recipe(self) -> tuple[str, float]:
        """Return (name, probability) of most likely recipe."""
        best = max(self.belief, key=self.belief.get)
        return best, self.belief[best]

    def ig_q(self) -> float | None:
        """IG from last clarification answer only."""
        if not self._ig_q_log:
            return None
        return self._ig_q_log[-1]["ig_q"]

    def ig_q_log(self) -> list[dict]:
        """Full log of all clarification IG events."""
        return list(self._ig_q_log)

    def current_similarities(self) -> dict[str, float]:
        """Raw hybrid similarity scores from last update."""
        return dict(self._last_similarities)

    def current_contrastive(self) -> dict[str, float]:
        """Contrastive similarity scores (raw minus mean) from last update."""
        return dict(self._last_contrastive)

    def reset(self):
        """Reset for a new session."""
        self._scene.reset()
        self.belief = {name: 1.0 / self.N for name in self.recipe_names}
        self.history = [dict(self.belief)]
        self._entropy_before_answer = None
        self._ig_q_log = []
        self._last_similarities = {}
        self._last_contrastive = {}

    def summary(self) -> dict:
        top, prob = self.top_recipe()
        return {
            "belief": dict(self.belief),
            "entropy": round(self.entropy(), 4),
            "max_entropy": round(self.max_entropy(), 4),
            "top_recipe": top,
            "top_prob": round(prob, 4),
            "ig_q": self.ig_q(),
            "temperature": self.temperature,
            "sequence_length": self._scene.current_length(),
            "observations": self._scene.summary()["observations"],
        }

    def copy(self) -> "BeliefUpdaterV3":
        """
        Create a lightweight copy for EIG simulation.
        Shares embedder and recipe sequences (read-only).
        Does NOT share observation sequence or belief state.
        """
        clone = BeliefUpdaterV3.__new__(BeliefUpdaterV3)
        clone.temperature = self.temperature
        clone.recipe_sequences = self.recipe_sequences
        clone.recipe_names = self.recipe_names
        clone.N = self.N
        clone._embedder = self._embedder
        clone._scene = DynamicScene(embedder=self._embedder)

        # Copy current observation sequence
        for meta in self._scene.get_sentences():
            if meta["type"] == "observation":
                clone._scene.add(meta["sentence"])
            else:
                clone._scene.add_answer(meta["sentence"])

        clone.belief = dict(self.belief)
        clone.history = [dict(self.belief)]
        clone._entropy_before_answer = None
        clone._ig_q_log = []
        clone._last_similarities = dict(self._last_similarities)
        clone._last_contrastive = dict(self._last_contrastive)
        return clone

    # ── Internal ───────────────────────────────────────────────────────────

    def _recompute(self) -> dict[str, float]:
        """
        Recompute full belief distribution from current observation sequence.

        For each recipe:
          1. Compute hybrid similarity (F1 + order consistency)
          2. Subtract mean similarity (contrastive normalisation)
          3. Apply softmax with fixed temperature
          4. Normalise to probability distribution

        Contrastive normalisation:
          contrastive_sim(r) = hybrid_sim(r) - mean(hybrid_sim over all recipes)

          When all recipes match equally (generic observation like "pours water"),
          all contrastive similarities are zero → softmax stays uniform → no update.
          When one recipe matches distinctively better, its positive contrast
          drives probability mass toward it.
        """
        obs_seq = self._scene.get_sequence()

        if not obs_seq:
            return dict(self.belief)

        # Hybrid similarity for each recipe
        similarities = np.array([
            hybrid_similarity(obs_seq, self.recipe_sequences[name])
            for name in self.recipe_names
        ], dtype=np.float32)

        self._last_similarities = {
            name: round(float(similarities[i]), 4)
            for i, name in enumerate(self.recipe_names)
        }

        # Contrastive normalisation — subtract mean
        contrastive = similarities - similarities.mean()

        self._last_contrastive = {
            name: round(float(contrastive[i]), 4)
            for i, name in enumerate(self.recipe_names)
        }

        # Softmax with fixed temperature
        probs = self._softmax(contrastive, self.temperature)

        # Build belief dict
        self.belief = {
            name: float(probs[i])
            for i, name in enumerate(self.recipe_names)
        }
        self.history.append(dict(self.belief))
        return dict(self.belief)

    @staticmethod
    def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
        scaled = values / temperature
        scaled -= scaled.max()
        exp_vals = np.exp(scaled)
        return exp_vals / exp_vals.sum()