"""
belief_updater.py
-----------------
Probabilistic belief updater over candidate recipes.

Combines:
  - IDF weighting: rare ingredients/actions across recipes get higher weight
  - Stage-aware matching: rewards observations that match the expected
    current stage of a recipe, penalises contradictions
  - Bayesian update: P_t(r) ∝ P_{t-1}(r) * L(obs | r)
  - Normalisation: belief always sums to 1.0

Usage:
    from belief_updater import BeliefUpdater
    from test_recipes import RECIPES
    from test_scene_graph import build_mock_session

    updater = BeliefUpdater(RECIPES)
    for history in build_mock_session():
        belief = updater.update(history)
        print(belief)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from src.dynamic_graph import ObservationHistory


# ═══════════════════════════════════════════════════════════════════════════
# IDF computer
# ═══════════════════════════════════════════════════════════════════════════

class IDFIndex:
    """
    Computes inverse document frequency weights for ingredients and actions
    across the full recipe set.

    IDF(term) = log(N / df(term)) + 1
    where N = number of recipes, df(term) = number of recipes containing term.

    Common terms (cucumber, onion) get low weight.
    Rare terms (sour cream, celery seed) get high weight.
    """

    def __init__(self, recipes: list[dict]):
        self.N = len(recipes)
        self._ingredient_df: dict[str, int] = defaultdict(int)
        self._action_df: dict[str, int] = defaultdict(int)
        self._build(recipes)

    def _build(self, recipes: list[dict]):
        for recipe in recipes:
            # Ingredient document frequency
            ing_set = set(recipe.get("all_ingredients", []))
            for ing in ing_set:
                self._ingredient_df[ing] += 1

            # Action document frequency — collect all actions across all steps
            action_set = set()
            for stage in recipe.get("stages", []):
                for step in stage.get("steps", []):
                    for action in step.get("actions", []):
                        action_set.add(action)
            for action in action_set:
                self._action_df[action] += 1

    def ingredient_weight(self, ingredient: str) -> float:
        df = self._ingredient_df.get(ingredient, 0)
        if df == 0:
            return math.log(self.N) + 1  # unseen ingredient → max weight
        return math.log(self.N / df) + 1

    def action_weight(self, action: str) -> float:
        df = self._action_df.get(action, 0)
        if df == 0:
            return math.log(self.N) + 1
        return math.log(self.N / df) + 1


# ═══════════════════════════════════════════════════════════════════════════
# Stage tracker
# ═══════════════════════════════════════════════════════════════════════════

class StageTracker:
    """
    Tracks which stage a recipe is currently at given observations so far.

    A stage is considered 'active' when at least one ingredient OR one action
    from that stage has been observed. Stages must be completed in order —
    you cannot be in Stage 2 before Stage 1 is active.
    """

    def __init__(self, recipe: dict):
        self.stages = recipe.get("stages", [])

    def current_stage_index(self, ingredients_seen: set[str], actions_seen: set[str]) -> int:
        """
        Returns the index of the current stage (0-based).
        Advances through stages in order as observations accumulate.
        """
        current = 0
        for i, stage in enumerate(self.stages):
            stage_ingredients = set()
            stage_actions = set()
            for step in stage.get("steps", []):
                stage_ingredients.update(step.get("ingredients", []))
                stage_actions.update(step.get("actions", []))

            # Stage is active if at least one ingredient OR action is observed
            ing_match = bool(stage_ingredients & ingredients_seen)
            act_match = bool(stage_actions & actions_seen)

            if ing_match or act_match:
                current = i  # advance to this stage
            else:
                break  # stop — don't skip stages

        return current

    def get_stage(self, index: int) -> Optional[dict]:
        if 0 <= index < len(self.stages):
            return self.stages[index]
        return None

    def get_expected_ingredients(self, stage_index: int) -> set[str]:
        stage = self.get_stage(stage_index)
        if not stage:
            return set()
        result = set()
        for step in stage.get("steps", []):
            result.update(step.get("ingredients", []))
        return result

    def get_expected_actions(self, stage_index: int) -> set[str]:
        stage = self.get_stage(stage_index)
        if not stage:
            return set()
        result = set()
        for step in stage.get("steps", []):
            result.update(step.get("actions", []))
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Likelihood function
# ═══════════════════════════════════════════════════════════════════════════

def compute_likelihood(
    recipe: dict,
    ingredients_seen: set[str],
    actions_seen: set[str],
    idf: IDFIndex,
    stage_tracker: StageTracker,
) -> float:
    """
    Compute L(observations | recipe) combining:
      1. IDF-weighted ingredient overlap
      2. IDF-weighted action overlap
      3. Stage consistency bonus
      4. Contradiction penalty
    Returns a positive float (not normalised).
    """
    all_recipe_ingredients = set(recipe.get("all_ingredients", []))

    # ── 1. IDF-weighted ingredient overlap ────────────────────────────────
    ing_score = 0.0
    ing_norm = 0.0
    for ing in all_recipe_ingredients:
        w = idf.ingredient_weight(ing)
        ing_norm += w
        if ing in ingredients_seen:
            ing_score += w
    ing_score = (ing_score / ing_norm) if ing_norm > 0 else 0.0

    # ── 2. IDF-weighted action overlap ────────────────────────────────────
    all_recipe_actions: set[str] = set()
    for stage in recipe.get("stages", []):
        for step in stage.get("steps", []):
            all_recipe_actions.update(step.get("actions", []))

    act_score = 0.0
    act_norm = 0.0
    for act in all_recipe_actions:
        w = idf.action_weight(act)
        act_norm += w
        if act in actions_seen:
            act_score += w
    act_score = (act_score / act_norm) if act_norm > 0 else 0.0

    # ── 3. Stage consistency bonus ────────────────────────────────────────
    current_stage_idx = stage_tracker.current_stage_index(ingredients_seen, actions_seen)
    expected_ingredients = stage_tracker.get_expected_ingredients(current_stage_idx)
    expected_actions = stage_tracker.get_expected_actions(current_stage_idx)

    stage_bonus = 0.0
    latest_ingredients = ingredients_seen  # latest window ingredients
    latest_actions = actions_seen

    if expected_ingredients and (expected_ingredients & latest_ingredients):
        stage_bonus += 0.2
    if expected_actions and (expected_actions & latest_actions):
        stage_bonus += 0.2

    # ── 4. Contradiction penalty ──────────────────────────────────────────
    # Ingredients observed that don't appear anywhere in this recipe
    contradicting = ingredients_seen - all_recipe_ingredients
    penalty = 0.0
    if contradicting and ingredients_seen:
        # Weight contradictions by their IDF — a rare ingredient seen
        # that doesn't belong here is a stronger contradiction
        contradiction_weight = sum(idf.ingredient_weight(i) for i in contradicting)
        total_weight = sum(idf.ingredient_weight(i) for i in ingredients_seen)
        penalty = 0.3 * (contradiction_weight / total_weight)

    # ── Combine ───────────────────────────────────────────────────────────
    # Weighted combination — ingredients slightly more important than actions
    raw = (0.45 * ing_score) + (0.35 * act_score) + stage_bonus - penalty

    # Clamp to small positive value so no recipe ever gets probability 0
    return max(raw, 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Belief updater
# ═══════════════════════════════════════════════════════════════════════════

class BeliefUpdater:
    """
    Maintains P(recipe | observations) and updates it after each window.

    Parameters
    ----------
    recipes : list of recipe dicts (matching recipe graph JSON structure)
    """

    def __init__(self, recipes: list[dict]):
        self.recipes = recipes
        n = len(recipes)

        # Uniform prior
        self.belief: dict[str, float] = {r["name"]: 1.0 / n for r in recipes}
        self.history: list[dict[str, float]] = [dict(self.belief)]

        # Build IDF index once across all recipes
        self.idf = IDFIndex(recipes)

        # One stage tracker per recipe
        self.stage_trackers: dict[str, StageTracker] = {
            r["name"]: StageTracker(r) for r in recipes
        }

    def update(self, obs_history: ObservationHistory) -> dict[str, float]:
        """
        Update belief given the full observation history so far.
        Returns the new normalised belief distribution.
        """
        ingredients_seen = set(obs_history.ingredients_seen())
        actions_seen = set(obs_history.actions_so_far())

        # Compute likelihood for each recipe
        likelihoods: dict[str, float] = {}
        for recipe in self.recipes:
            name = recipe["name"]
            tracker = self.stage_trackers[name]
            likelihoods[name] = compute_likelihood(
                recipe, ingredients_seen, actions_seen, self.idf, tracker
            )

        # Bayesian update: P_t(r) ∝ P_{t-1}(r) * L(obs | r)
        new_belief: dict[str, float] = {}
        for name in self.belief:
            new_belief[name] = self.belief[name] * likelihoods[name]

        # Normalise so probabilities sum to 1
        total = sum(new_belief.values())
        if total > 0:
            new_belief = {k: v / total for k, v in new_belief.items()}
        else:
            # Fallback to uniform if all likelihoods collapse
            n = len(self.recipes)
            new_belief = {r["name"]: 1.0 / n for r in self.recipes}

        self.belief = new_belief
        self.history.append(dict(self.belief))
        return dict(self.belief)

    def entropy(self) -> float:
        """Shannon entropy of current belief in bits."""
        return -sum(
            p * math.log2(p) for p in self.belief.values() if p > 0
        )

    def max_entropy(self) -> float:
        return math.log2(len(self.recipes))

    def top_recipe(self) -> tuple[str, float]:
        best = max(self.belief, key=self.belief.get)
        return best, self.belief[best]

    def should_ask(self, threshold: float = 1.5) -> bool:
        """Returns True if entropy exceeds threshold — time to ask a question."""
        return self.entropy() > threshold

    def reset(self):
        n = len(self.recipes)
        self.belief = {r["name"]: 1.0 / n for r in self.recipes}
        self.history = [dict(self.belief)]

    def summary(self) -> dict:
        top, prob = self.top_recipe()
        return {
            "belief": dict(self.belief),
            "entropy": round(self.entropy(), 4),
            "max_entropy": round(self.max_entropy(), 4),
            "top_recipe": top,
            "top_prob": round(prob, 4),
            "should_ask": self.should_ask(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Main — run full test session and print results
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from test_recipes import RECIPES
    from test_scene_graph import build_mock_session

    print("=== Belief Updater Test ===")
    print(f"Recipes: {[r['name'] for r in RECIPES]}")
    print(f"Max entropy (uniform over {len(RECIPES)}): {math.log2(len(RECIPES)):.3f} bits\n")

    updater = BeliefUpdater(RECIPES)
    histories = build_mock_session()

    for i, history in enumerate(histories, start=1):
        belief = updater.update(history)
        s = updater.summary()

        print(f"── Window {i} ──────────────────────────────────────────")
        print(f"  ingredients_seen : {history.ingredients_seen()}")
        print(f"  actions_so_far   : {history.actions_so_far()}")
        print(f"  entropy          : {s['entropy']:.3f} / {s['max_entropy']:.3f} bits")
        print(f"  should ask?      : {s['should_ask']}")
        print(f"  top recipe       : {s['top_recipe']} ({s['top_prob']:.3f})")
        print()
        print("  Belief distribution:")
        for name, prob in sorted(belief.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 40)
            print(f"    {name:35s} {prob:.4f}  {bar}")
        print()

    print("═" * 60)
    print(f"Final prediction : {updater.top_recipe()[0]}")
    print(f"Correct answer   : creamy cucumber salad")
    print(f"Correct?         : {updater.top_recipe()[0] == 'creamy cucumber salad'}")
