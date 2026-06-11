"""
belief_updater_v2.py
--------------------
Embedding-based belief updater using bidirectional F1-style matching.

Changes in this version:
- Compound term extraction in incorporate_answer
  (matches answer bigrams/trigrams against known recipe terms)
- F1 similarity scores exposed per update for debugging
"""

from __future__ import annotations

import json
import math
import numpy as np
from embedder import Embedder


RECIPE_TERMS_PATH = "recipe_terms.json"
SIMILARITY_THRESHOLD = 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Cosine similarity
# ═══════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ═══════════════════════════════════════════════════════════════════════════
# Compound term extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_compound_terms(
    answer_text: str,
    known_recipe_terms: set[str],
) -> list[str]:
    """
    Extract meaningful terms from answer text by matching against
    known recipe terms vocabulary.

    Strategy:
    1. Generate all unigrams, bigrams, trigrams from answer
    2. Keep only those that match a known recipe term
    3. Prefer longer matches (trigram > bigram > unigram)
    4. Filter stopwords from unigrams

    Parameters
    ----------
    answer_text       : natural language answer from human
    known_recipe_terms: set of all terms across all recipes

    Returns list of matched terms, compound ones kept intact.
    """
    stopwords = {
        "i", "am", "is", "are", "the", "a", "an", "to", "and", "or",
        "of", "in", "it", "my", "me", "we", "be", "some", "adding",
        "using", "with", "have", "just", "now", "going", "use", "put",
        "this", "that", "there", "here", "its"
    }

    # Clean and tokenise
    words = [
        w.strip(".,!?\"'").lower()
        for w in answer_text.split()
    ]
    words = [w for w in words if w]

    # Generate ngrams
    ngrams: list[str] = []
    n = len(words)

    # Trigrams
    for i in range(n - 2):
        ngrams.append(" ".join(words[i:i+3]))
    # Bigrams
    for i in range(n - 1):
        ngrams.append(" ".join(words[i:i+2]))
    # Unigrams (filter stopwords)
    for w in words:
        if w not in stopwords and len(w) > 2:
            ngrams.append(w)

    # Match against known recipe terms
    # Prefer longer matches — track which word positions are already covered
    matched: list[str] = []
    covered_positions: set[int] = set()

    # Sort by length descending so longer matches win
    for ngram in sorted(ngrams, key=lambda x: len(x.split()), reverse=True):
        if ngram in known_recipe_terms:
            # Find word positions this ngram covers
            ngram_words = ngram.split()
            ngram_len = len(ngram_words)
            for i in range(n - ngram_len + 1):
                if words[i:i+ngram_len] == ngram_words:
                    positions = set(range(i, i + ngram_len))
                    # Only add if positions not already covered
                    if not positions & covered_positions:
                        matched.append(ngram)
                        covered_positions.update(positions)
                    break

    # Also embed the full answer as one term for broader semantic coverage
    matched.append(answer_text.strip())

    return list(set(matched))


# ═══════════════════════════════════════════════════════════════════════════
# F1 bidirectional matching
# ═══════════════════════════════════════════════════════════════════════════

def f1_match(
    obs_term_vectors: dict[str, np.ndarray],
    recipe_term_vectors: dict[str, np.ndarray],
    threshold: float = SIMILARITY_THRESHOLD,
) -> float:
    """
    Bidirectional F1-style similarity between observed terms and recipe terms.

    Coverage  (recall):  how well recipe terms are covered by observations.
    Precision:           how well observed terms match recipe terms.
    F1:                  harmonic mean of coverage and precision.
    """
    if not obs_term_vectors or not recipe_term_vectors:
        return 0.0

    obs_vecs = list(obs_term_vectors.values())
    rec_vecs = list(recipe_term_vectors.values())

    # Coverage: for each recipe term, best match in observations
    coverage_scores = []
    for rec_vec in rec_vecs:
        best = max(cosine_similarity(rec_vec, obs_vec) for obs_vec in obs_vecs)
        coverage_scores.append(best if best >= threshold else 0.0)
    coverage = float(np.mean(coverage_scores))

    # Precision: for each observed term, best match in recipe
    precision_scores = []
    for obs_vec in obs_vecs:
        best = max(cosine_similarity(obs_vec, rec_vec) for rec_vec in rec_vecs)
        precision_scores.append(best if best >= threshold else 0.0)
    precision = float(np.mean(precision_scores))

    if coverage + precision == 0:
        return 0.0
    return 2 * coverage * precision / (coverage + precision)


# ═══════════════════════════════════════════════════════════════════════════
# Belief updater v2
# ═══════════════════════════════════════════════════════════════════════════

class BeliefUpdaterV2:
    """
    F1-matching embedding-based belief updater with compound term extraction.

    Parameters
    ----------
    recipe_terms_path : path to recipe_terms.json
    temperature       : softmax temperature (lower = sharper distribution)
    threshold         : minimum cosine similarity to count as a match
    """

    def __init__(
        self,
        recipe_terms_path: str = RECIPE_TERMS_PATH,
        temperature: float = 0.1,
        threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.temperature = temperature
        self.threshold = threshold

        # Load recipe terms
        with open(recipe_terms_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.recipe_term_vectors: dict[str, dict[str, np.ndarray]] = {
            name: {
                term: np.array(vec, dtype=np.float32)
                for term, vec in terms.items()
            }
            for name, terms in raw.items()
        }

        self.recipe_names = list(self.recipe_term_vectors.keys())
        self.N = len(self.recipe_names)

        # Build flat set of all known recipe terms for compound matching
        self.known_recipe_terms: set[str] = set()
        for terms in self.recipe_term_vectors.values():
            self.known_recipe_terms.update(terms.keys())

        self.embedder = Embedder()

        # Uniform prior
        self.belief: dict[str, float] = {
            name: 1.0 / self.N for name in self.recipe_names
        }
        self.history: list[dict[str, float]] = [dict(self.belief)]

        # Accumulated observation terms
        self._obs_terms: dict[str, np.ndarray] = {}

        # Entropy + similarity tracking
        self._entropy_before_last_answer: float | None = None
        self._entropy_after_last_answer: float | None = None
        self._ig_q_history: list[dict] = []
        self._last_similarities: dict[str, float] = {}

        print(f"[BeliefUpdaterV2] Loaded {self.N} recipes.")
        print(f"[BeliefUpdaterV2] Known recipe terms: {len(self.known_recipe_terms)}")
        print(f"[BeliefUpdaterV2] Temperature: {self.temperature} | Threshold: {self.threshold}")

    # ── Public API ─────────────────────────────────────────────────────────

    def update(
        self,
        ingredients_seen: list[str],
        actions_seen: list[str],
        object_states: dict[str, str] = None,
    ) -> dict[str, float]:
        """Update belief after a new observation window."""
        all_terms = list(ingredients_seen) + list(actions_seen)
        new_terms = [t for t in all_terms if t not in self._obs_terms]
        if new_terms:
            new_vectors = self.embedder.embed_batch(new_terms)
            for term, vec in zip(new_terms, new_vectors):
                self._obs_terms[term] = vec
        return self._recompute_belief()

    def incorporate_answer(self, answer_text: str) -> dict[str, float]:
        """
        Incorporate a clarification answer into the belief.
        Extracts compound terms by matching against known recipe vocabulary.
        """
        self._entropy_before_last_answer = self.entropy()

        # Extract compound terms matched against recipe vocabulary
        matched_terms = extract_compound_terms(
            answer_text, self.known_recipe_terms
        )

        print(f"  [compound terms extracted]: {matched_terms}")

        # Embed matched terms
        new_terms = [t for t in matched_terms if t not in self._obs_terms]
        if new_terms:
            new_vectors = self.embedder.embed_batch(new_terms)
            for term, vec in zip(new_terms, new_vectors):
                self._obs_terms[term] = vec

        result = self._recompute_belief()

        # Snapshot entropy immediately after answer
        self._entropy_after_last_answer = self.entropy()
        ig = self._entropy_before_last_answer - self._entropy_after_last_answer
        self._ig_q_history.append({
            "answer": answer_text,
            "extracted_terms": matched_terms,
            "entropy_before": round(self._entropy_before_last_answer, 4),
            "entropy_after": round(self._entropy_after_last_answer, 4),
            "ig_q": round(ig, 4),
        })

        return result

    def entropy(self) -> float:
        return -sum(
            p * math.log2(p)
            for p in self.belief.values()
            if p > 0
        )

    def max_entropy(self) -> float:
        return math.log2(self.N)

    def ig_q(self) -> float | None:
        """IG from last clarification answer only — not affected by subsequent obs."""
        if not self._ig_q_history:
            return None
        return self._ig_q_history[-1]["ig_q"]

    def ig_q_history(self) -> list[dict]:
        return list(self._ig_q_history)

    def last_similarities(self) -> dict[str, float]:
        """F1 similarity scores from the last update."""
        return dict(self._last_similarities)

    def top_recipe(self) -> tuple[str, float]:
        best = max(self.belief, key=self.belief.get)
        return best, self.belief[best]

    def reset(self):
        self.belief = {name: 1.0 / self.N for name in self.recipe_names}
        self.history = [dict(self.belief)]
        self._obs_terms = {}
        self._entropy_before_last_answer = None
        self._entropy_after_last_answer = None
        self._ig_q_history = []
        self._last_similarities = {}

    def summary(self) -> dict:
        top, prob = self.top_recipe()
        return {
            "belief": dict(self.belief),
            "entropy": round(self.entropy(), 4),
            "max_entropy": round(self.max_entropy(), 4),
            "top_recipe": top,
            "top_prob": round(prob, 4),
            "ig_q": round(self.ig_q(), 4) if self.ig_q() is not None else None,
            "obs_terms": list(self._obs_terms.keys()),
            "similarities": self._last_similarities,
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _recompute_belief(self) -> dict[str, float]:
        if not self._obs_terms:
            return dict(self.belief)

        # Compute F1 similarity for each recipe
        similarities: dict[str, float] = {}
        for name in self.recipe_names:
            similarities[name] = f1_match(
                self._obs_terms,
                self.recipe_term_vectors[name],
                threshold=self.threshold,
            )
        self._last_similarities = {
            k: round(v, 4) for k, v in similarities.items()
        }

        # Softmax with temperature
        sim_values = np.array([similarities[n] for n in self.recipe_names])
        softmax_vals = self._softmax(sim_values, self.temperature)
        softmax_sims = {
            name: float(softmax_vals[i])
            for i, name in enumerate(self.recipe_names)
        }

        # Bayesian update
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
        scaled = values / temperature
        scaled -= scaled.max()
        exp_vals = np.exp(scaled)
        return exp_vals / exp_vals.sum()