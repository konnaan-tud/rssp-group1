"""
belief_updater_v2.py
--------------------
Embedding-based probabilistic belief updater over candidate recipes.

Replaces the IDF-based belief_updater.py with:
  - Sentence transformer embeddings (all-mpnet-base-v2)
  - Combined cosine + euclidean similarity as likelihood
  - Bayesian update: P_t(r) ∝ P_{t-1}(r) × similarity(obs, recipe_r)
  - Softmax with temperature for probability conversion
  - Entropy and IG_Q measurement

Usage:
    from belief_updater_v2 import BeliefUpdaterV2
    updater = BeliefUpdaterV2()
    updater.update(ingredients_seen, actions_seen, object_states)
    updater.incorporate_answer("I am adding sour cream")
    print(updater.entropy())
"""

from __future__ import annotations

import json
import math
import numpy as np
from embedder import Embedder

VECTORS_PATH = "recipe_vectors.npy"
INDEX_PATH = "recipe_index.json"


# ═══════════════════════════════════════════════════════════════════════════
# Similarity functions
# ═══════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance converted to similarity in [0, 1].
    sim = 1 / (1 + distance)
    """
    dist = float(np.linalg.norm(a - b))
    return 1.0 / (1.0 + dist)


def combined_similarity(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.7,
) -> float:
    """
    Combined cosine + euclidean similarity.
    alpha controls weight of cosine (1-alpha for euclidean).
    Returns value in [0, 1].
    """
    cos = cosine_similarity(a, b)
    # Normalise cosine from [-1,1] to [0,1]
    cos_norm = (cos + 1.0) / 2.0
    euc = euclidean_similarity(a, b)
    return alpha * cos_norm + (1.0 - alpha) * euc


# ═══════════════════════════════════════════════════════════════════════════
# Observation text builder
# ═══════════════════════════════════════════════════════════════════════════

def build_observation_text(
    ingredients_seen: list[str],
    actions_seen: list[str],
    object_states: dict[str, str],
) -> str:
    """
    Convert accumulated observations into structured natural text for embedding.

    Format:
        "observed ingredients: <ing1>, <ing2>.
         observed actions: <act1>, <act2>.
         object states: <obj> is <state>, ..."
    """
    parts = []

    if ingredients_seen:
        parts.append(f"observed ingredients: {', '.join(ingredients_seen)}")

    if actions_seen:
        parts.append(f"observed actions: {', '.join(actions_seen)}")

    if object_states:
        state_parts = [
            f"{obj.replace('_', ' ')} is {state.replace('_', ' ')}"
            for obj, state in object_states.items()
        ]
        parts.append(f"object states: {', '.join(state_parts)}")

    return ". ".join(parts) + "." if parts else ""


# ═══════════════════════════════════════════════════════════════════════════
# Belief updater v2
# ═══════════════════════════════════════════════════════════════════════════

class BeliefUpdaterV2:
    """
    Embedding-based belief updater over candidate recipes.

    Parameters
    ----------
    vectors_path : path to recipe_vectors.npy
    index_path   : path to recipe_index.json
    alpha        : cosine weight in combined similarity (0.7 default)
    temperature  : softmax temperature (0.5 default)
    """

    def __init__(
        self,
        vectors_path: str = VECTORS_PATH,
        index_path: str = INDEX_PATH,
        alpha: float = 0.7,
        temperature: float = 0.5,
    ):
        self.alpha = alpha
        self.temperature = temperature

        # Load recipe vectors and index
        self.recipe_vectors = np.load(vectors_path)  # shape (N, 768)
        with open(index_path, "r", encoding="utf-8") as f:
            self.recipe_index = json.load(f)  # name → row index

        self.recipe_names = list(self.recipe_index.keys())
        self.N = len(self.recipe_names)

        # Embedder
        self.embedder = Embedder()

        # Uniform prior
        self.belief: dict[str, float] = {
            name: 1.0 / self.N for name in self.recipe_names
        }
        self.history: list[dict[str, float]] = [dict(self.belief)]

        # Accumulated observation state
        self._ingredients_seen: list[str] = []
        self._actions_seen: list[str] = []
        self._object_states: dict[str, str] = {}
        self._answer_texts: list[str] = []

        # Current observation vector
        self._obs_vector: np.ndarray | None = None

        # Entropy tracking for IG_Q
        self._entropy_before_last_answer: float | None = None

        print(f"[BeliefUpdaterV2] Loaded {self.N} recipes.")

    # ── Public API ─────────────────────────────────────────────────────────

    def update(
        self,
        ingredients_seen: list[str],
        actions_seen: list[str],
        object_states: dict[str, str],
    ) -> dict[str, float]:
        """
        Update belief after a new observation window.

        Parameters
        ----------
        ingredients_seen : all ingredients observed so far (cumulative)
        actions_seen     : all actions observed so far (cumulative)
        object_states    : latest known states per object

        Returns updated belief distribution.
        """
        # Update accumulated state
        self._ingredients_seen = ingredients_seen
        self._actions_seen = actions_seen
        self._object_states.update(object_states)

        return self._recompute_belief()

    def incorporate_answer(self, answer_text: str) -> dict[str, float]:
        """
        Incorporate a clarification answer into the belief.

        Records entropy before update so IG_Q can be computed.

        Parameters
        ----------
        answer_text : natural language answer from human

        Returns updated belief distribution.
        """
        self._entropy_before_last_answer = self.entropy()
        self._answer_texts.append(answer_text)
        return self._recompute_belief()

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

    def ig_q(self) -> float | None:
        """
        Information gain from the last clarification answer.
        Returns None if no answer has been incorporated yet.
        """
        if self._entropy_before_last_answer is None:
            return None
        return self._entropy_before_last_answer - self.entropy()

    def top_recipe(self) -> tuple[str, float]:
        """Return (recipe_name, probability) of most likely recipe."""
        best = max(self.belief, key=self.belief.get)
        return best, self.belief[best]

    def reset(self):
        """Reset for a new session."""
        self.belief = {name: 1.0 / self.N for name in self.recipe_names}
        self.history = [dict(self.belief)]
        self._ingredients_seen = []
        self._actions_seen = []
        self._object_states = {}
        self._answer_texts = []
        self._obs_vector = None
        self._entropy_before_last_answer = None

    def summary(self) -> dict:
        top, prob = self.top_recipe()
        return {
            "belief": dict(self.belief),
            "entropy": round(self.entropy(), 4),
            "max_entropy": round(self.max_entropy(), 4),
            "top_recipe": top,
            "top_prob": round(prob, 4),
            "ig_q": round(self.ig_q(), 4) if self.ig_q() is not None else None,
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _build_obs_text(self) -> str:
        """
        Build full observation text from accumulated state + answers.
        Re-built from scratch each update (Option A).
        """
        parts = []

        obs_text = build_observation_text(
            self._ingredients_seen,
            self._actions_seen,
            self._object_states,
        )
        if obs_text:
            parts.append(obs_text)

        # Append clarification answers
        for answer in self._answer_texts:
            parts.append(f"clarification: {answer}")

        return " ".join(parts)

    def _recompute_belief(self) -> dict[str, float]:
        """
        Core update: embed current observation, compute similarities,
        Bayesian update, normalise.
        """
        obs_text = self._build_obs_text()

        if not obs_text.strip():
            return dict(self.belief)

        # Embed full observation text
        self._obs_vector = self.embedder.embed(obs_text)

        # Compute combined similarity against every recipe vector
        similarities: dict[str, float] = {}
        for name in self.recipe_names:
            idx = self.recipe_index[name]
            recipe_vec = self.recipe_vectors[idx]
            similarities[name] = combined_similarity(
                self._obs_vector, recipe_vec, alpha=self.alpha
            )

        # Apply softmax with temperature to similarities
        sim_values = np.array([similarities[n] for n in self.recipe_names])
        softmax_vals = self._softmax(sim_values, self.temperature)
        softmax_sims = {
            name: float(softmax_vals[i])
            for i, name in enumerate(self.recipe_names)
        }

        # Bayesian update: P_t(r) ∝ P_{t-1}(r) × softmax_sim(r)
        new_belief: dict[str, float] = {}
        for name in self.recipe_names:
            new_belief[name] = self.belief[name] * softmax_sims[name]

        # Normalise
        total = sum(new_belief.values())
        if total > 0:
            new_belief = {k: v / total for k, v in new_belief.items()}
        else:
            new_belief = {name: 1.0 / self.N for name in self.recipe_names}

        self.belief = new_belief
        self.history.append(dict(self.belief))
        return dict(self.belief)

    @staticmethod
    def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
        """Softmax with temperature."""
        scaled = values / temperature
        # Subtract max for numerical stability
        scaled -= scaled.max()
        exp_vals = np.exp(scaled)
        return exp_vals / exp_vals.sum()