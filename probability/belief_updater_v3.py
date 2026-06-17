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
RECIPE_SCENES_PATH = os.path.join("data", "italian_recipe_scenes.json")

# Fixed penalty strength for a "no" answer in the polar condition. Held
# constant across ALL polar sessions for the same reason temperature is
# fixed: IG values must stay on one scale to be comparable. Subtracted from
# a recipe's raw hybrid similarity (before the contrastive mean) in
# proportion to how strongly that recipe's near-future matches the negated
# proposition. Tuned once against the carbonara clips, then frozen.
NEGATION_PENALTY = 0.5
NEGATION_MATCH_THRESHOLD = 0.5   # cosine below this = recipe unaffected by the "no"


class BeliefUpdaterV3:
    """
    Contrastive hybrid similarity belief updater over candidate recipes.

    Parameters
    ----------
    recipe_sequences_path : path to data/recipe_sequences.json
                            (pre-embedded per-scene vectors)
    recipe_scenes_path    : path to data/italian_recipe_scenes.json
                            (raw scene sentences — used by the questioning
                            pipeline to show "next likely scenes" to the VLM
                            and by the planner to simulate hypothetical
                            answers per recipe)
    temperature           : softmax temperature (lower = sharper distribution)
    """

    def __init__(
        self,
        recipe_sequences_path: str = RECIPE_SEQUENCES_PATH,
        recipe_scenes_path: str = RECIPE_SCENES_PATH,
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

        # Load matching raw scene sentences. The file is a list of
        # {"dish": ..., "scenes": [...]} entries.
        self.recipe_sentences: dict[str, list[str]] = {}
        if os.path.exists(recipe_scenes_path):
            with open(recipe_scenes_path, "r", encoding="utf-8") as f:
                scenes_raw = json.load(f)
            for entry in scenes_raw:
                dish = entry.get("dish", "").strip()
                scenes = [s.strip() for s in entry.get("scenes", []) if s.strip()]
                if dish and scenes:
                    self.recipe_sentences[dish] = scenes

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

        # Persistent negative evidence: scene propositions the human said
        # "no" to (polar condition). Unlike self.belief, these survive every
        # recompute because they live in the evidence, not the distribution —
        # so a "no" is not wiped out by the next observation. The penalty is
        # re-applied inside _recompute() each step.
        self._negations: list[str] = []
        # Cache of negation embeddings (parallel to self._negations) so the
        # penalty loop doesn't re-embed every recompute.
        self._negation_vecs: list[np.ndarray] = []

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

    def incorporate_negative_answer(self, proposition: str) -> dict[str, float]:
        """
        Incorporate a polar "no": the human denied `proposition` (a scene-style
        statement, e.g. "A cook adds cream to the pan").

        Unlike incorporate_answer, the proposition is NOT appended to the
        observation sequence — a denial is not an observation. Instead it is
        stored in self._negations and re-applied as a similarity penalty on
        every subsequent recompute (see _recompute). IG_Q is logged the same
        way so polar "no" answers produce a measurable dependent variable.
        """
        self._entropy_before_answer = self.entropy()

        self._negations.append(proposition)
        self._negation_vecs.append(self._embedder.embed(proposition))
        result = self._recompute()

        entropy_after = self.entropy()
        ig = self._entropy_before_answer - entropy_after
        self._ig_q_log.append({
            "answer": f"NO: {proposition}",
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
        self._negations = []
        self._negation_vecs = []

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

    def observed_sentences(self) -> list[str]:
        """Strings of all observations (clips + answers) added so far, in order."""
        return [meta["sentence"] for meta in self._scene.get_sentences()]

    def unseen_recipe_scenes(
        self,
        recipe_name: str,
        cosine_threshold: float = 0.75,
    ) -> list[str]:
        """
        Recipe scene sentences (from italian_recipe_scenes.json) for
        `recipe_name` that have not yet been observed.

        A scene is considered "observed" if SOME sentence in the observation
        sequence has cosine similarity ≥ cosine_threshold to it. This is the
        semantic-match version — earlier this used literal string equality,
        which failed when the VLM paraphrased ("cracks an egg into a small
        white cup" vs the recipe's "cracks eggs into a mixing bowl") or
        when an answer used first person ("I am grating..." vs "A cook
        grates...").

        Used by:
          - the questioning prompt builder (next-likely-scenes context)
          - the planner simulator (hypothetical per-recipe answer)
        """
        scenes = self.recipe_sentences.get(recipe_name, [])
        if not scenes:
            return []

        observed = self.observed_sentences()
        if not observed:
            return list(scenes)

        # Embed both sets — cached, so this is essentially free after the
        # first lookup.
        scene_vecs = self._embedder.embed_batch(scenes)
        obs_vecs = self._embedder.embed_batch(observed)

        unseen: list[str] = []
        for scene, s_vec in zip(scenes, scene_vecs):
            # Highest cosine of this recipe scene against any observed sentence
            sn = float(np.linalg.norm(s_vec))
            best = 0.0
            if sn > 0.0:
                for o_vec in obs_vecs:
                    on = float(np.linalg.norm(o_vec))
                    if on == 0.0:
                        continue
                    c = float(np.dot(s_vec, o_vec) / (sn * on))
                    if c > best:
                        best = c
            if best < cosine_threshold:
                unseen.append(scene)
        return unseen

    def copy(self) -> "BeliefUpdaterV3":
        """
        Create a lightweight copy for EIG simulation.
        Shares embedder and recipe sequences (read-only).
        Does NOT share observation sequence or belief state.
        """
        clone = BeliefUpdaterV3.__new__(BeliefUpdaterV3)
        clone.temperature = self.temperature
        clone.recipe_sequences = self.recipe_sequences
        clone.recipe_sentences = self.recipe_sentences
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
        clone._negations = list(self._negations)
        clone._negation_vecs = list(self._negation_vecs)
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

        # Negative evidence (polar "no"): for each negated proposition, pull
        # down every recipe whose near-future scenes match it. Applied to raw
        # similarities BEFORE the contrastive mean so the whole vector stays
        # on the fixed-temperature scale and IG remains comparable. Recorded
        # AFTER _last_similarities so the logged raw scores stay penalty-free.
        if self._negation_vecs:
            penalties = np.zeros(self.N, dtype=np.float32)
            for i, name in enumerate(self.recipe_names):
                unseen = self.unseen_recipe_scenes(name)
                if not unseen:
                    continue
                scene_vecs = self._embedder.embed_batch(unseen)
                worst = 0.0  # strongest match to ANY negated proposition
                for neg_vec in self._negation_vecs:
                    nn = float(np.linalg.norm(neg_vec))
                    if nn == 0.0:
                        continue
                    for sv in scene_vecs:
                        sn = float(np.linalg.norm(sv))
                        if sn == 0.0:
                            continue
                        c = float(np.dot(neg_vec, sv) / (nn * sn))
                        if c > worst:
                            worst = c
                if worst >= NEGATION_MATCH_THRESHOLD:
                    penalties[i] = NEGATION_PENALTY * worst
            similarities = similarities - penalties

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