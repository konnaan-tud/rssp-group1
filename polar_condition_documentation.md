# Polar Condition — Technical Documentation

This document covers all changes made to the V3 pipeline to support the polar (yes/no) clarification condition. It is organised by component, from the belief updater upward to the orchestrator.

---

## 1. Background and Motivation

The wh condition generates clarification questions whose answers are scene-style sentences (e.g. "A cook grates pecorino into the mixing bowl."). These are embedded and appended to the observation sequence as positive evidence, updating the belief distribution in the same way a video observation does.

The polar condition generates yes/no questions that assert a single proposition (e.g. "Are you adding pecorino next?"). A "yes" is semantically equivalent to observing the proposition, so it follows the same path as a wh answer. A "no" is fundamentally different — it is evidence *against* a scene, and embedding the denial directly ("No, I am not adding pecorino") would cause the embedder to pick up noun content rather than the negation, potentially boosting the wrong recipes. A dedicated negative evidence path is therefore required.

---

## 2. Changes to `probability/belief_updater_v3.py`

### 2.1 New constants

Added at module level, near the existing path constants:

```python
NEGATION_PENALTY = 0.5
NEGATION_MATCH_THRESHOLD = 0.5
```

`NEGATION_PENALTY` is the strength of the downward pull applied to a recipe whose near-future matches a denied proposition. `NEGATION_MATCH_THRESHOLD` is the minimum cosine similarity required before any penalty is applied. Both are held constant across all polar sessions for the same reason `temperature` is fixed: IG_Q values must remain on a comparable scale across steps and conditions.

### 2.2 New state fields (`__init__`)

```python
self._negations: list[str] = []
self._negation_vecs: list[np.ndarray] = []
```

`_negations` stores the scene-style proposition strings denied by the human. `_negation_vecs` stores their embeddings (parallel list), so the penalty loop in `_recompute()` does not re-embed on every call. Both lists grow monotonically during a session — a "no" is never retracted.

### 2.3 `incorporate_negative_answer(proposition: str)`

New public method, placed immediately after `incorporate_answer()`.

```python
def incorporate_negative_answer(self, proposition: str) -> dict[str, float]:
```

Behaviour:

1. Records `H_before` for IG_Q logging.
2. Appends `proposition` to `self._negations` and its embedding to `self._negation_vecs`.
3. Calls `self._recompute()`. The negation penalty (Section 2.4) is applied automatically inside the recompute.
4. Computes `ig = H_before - H_after` and appends a log entry to `self._ig_q_log` with `answer` prefixed by `"NO: "`.

The proposition is **not** appended to the observation sequence. A denial is not an observation — it is persistent negative evidence stored separately from the observation trajectory and re-applied on every subsequent recompute.

### 2.4 Penalty block inside `_recompute()`

Inserted after `self._last_similarities` is recorded and **before** the contrastive mean is computed. This placement is deliberate: the penalty modifies raw similarities before the mean subtraction, so the contrastive normalisation and softmax run on the already-penalised values. Applying the penalty after the mean would cause a secondary bias in the contrastive scores of all other recipes (the mean would shift), violating the fixed-temperature comparability invariant.

```python
if self._negation_vecs:
    penalties = np.zeros(self.N, dtype=np.float32)
    for i, name in enumerate(self.recipe_names):
        unseen = self.unseen_recipe_scenes(name)
        if not unseen:
            continue
        scene_vecs = self._embedder.embed_batch(unseen)
        worst = 0.0
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
```

For each recipe, the block finds the strongest cosine match between any negated proposition and any of that recipe's remaining (unseen) scenes. If this maximum cosine clears `NEGATION_MATCH_THRESHOLD`, the recipe's raw similarity is reduced by `NEGATION_PENALTY × cosine`. Recipes whose near-future does not resemble anything denied are unaffected.

Because `_recompute()` is called on every `update()` and `incorporate_answer()` call, the penalty is re-derived from scratch at every step. A "no" given at window 3 continues to penalise matching recipes at window 7. This is the key property: negative evidence is persistent without requiring a Bayesian carry-forward of the distribution itself.

`self._last_similarities` is recorded before the penalty is applied, preserving the raw similarity scores in the log as a diagnostic signal.

### 2.5 `reset()` update

```python
self._negations = []
self._negation_vecs = []
```

Both fields are cleared on reset so that a new session within the same updater instance does not inherit negations from a prior session.

### 2.6 `copy()` update

```python
clone._negations = list(self._negations)
clone._negation_vecs = list(self._negation_vecs)
```

This is the most critical change for correctness of EIG simulation. The planner creates lightweight clones of the belief updater to simulate hypothetical answers. If the negations were not copied, the clone would operate without awareness of prior "no" answers, causing the predicted EIG for polar questions to be systematically overestimated. With the copy in place, every simulated branch inherits the full negative evidence history.

---

## 3. Changes to `questioning/questioning_pipeline.py`

### 3.1 `build_question_prompt_polar(recipe_context: str) -> str`

New function, parallel to `build_question_prompt()`. The key structural difference is the `proposition` field in the JSON schema.

In the wh prompt, each question has a `targets` list of scene sentences that would serve as valid answers. In the polar prompt, each question has a single `proposition` — the one scene-style statement the question asserts. This proposition is what gets incorporated on a "yes" or negated on a "no".

The prompt instructs the VLM to generate exactly 3 questions covering distinct categories (ingredient, method, sequence), and critically, to aim for questions that **split** the active candidates: roughly half should answer "yes" and half "no". A question all candidates answer identically has zero EIG.

The `proposition` must be drawn from the remaining (unseen) scenes of an active candidate, not invented. The prompt enforces this explicitly because a proposition that doesn't correspond to any recipe's actual next steps cannot be incorporated meaningfully.

Schema excerpt:

```json
{
  "question":      "<yes/no question>",
  "question_form": "polar",
  "category":      "ingredient | method | sequence",
  "proposition":   "<scene sentence from a candidate's remaining scenes>",
  "distinguishes": ["<candidate A>", "<candidate B>"],
  "expected_information_gain_reason": "<one sentence>"
}
```

---

## 4. Changes to `questioning/questioning_planner.py`

### 4.1 `simulate_polar_answer(question, recipe_name, belief_updater, min_cosine)`

New function, parallel to `simulate_answer()`. Returns `"yes"`, `"no"`, or `None`.

For a given recipe and question, the function embeds the proposition and checks its cosine similarity against the recipe's adaptive near-future window (same window as `simulate_answer` uses). If the maximum cosine across near-future scenes clears `min_cosine` (default: `APPLICABLE_MIN_COSINE`), the recipe would answer "yes" — the proposition describes something in its immediate upcoming steps. Otherwise it answers "no".

`None` is returned when the recipe has no remaining unseen scenes or the proposition string is empty, meaning the branch contributes zero to EIG (same semantics as `simulate_answer` returning `None`).

### 4.2 `expected_information_gain()` — polar branch

The function signature gains a `polar: bool = False` parameter. Inside the per-recipe loop, the simulation and incorporation logic forks:

```python
if polar:
    ans = simulate_polar_answer(question, recipe_name, belief_updater)
else:
    ans = simulate_answer(question, recipe_name, belief_updater)

# ...

if polar:
    if ans == "yes":
        clone.incorporate_answer(question["proposition"])
    else:
        clone.incorporate_negative_answer(question["proposition"])
else:
    clone.incorporate_answer(ans)
```

A simulated "yes" calls `incorporate_answer` on the clone — the same positive path as a wh answer. A simulated "no" calls `incorporate_negative_answer` — the negation penalty path. Because the clone carries the full negation history from `copy()`, each simulated branch accurately reflects what the belief would look like after that answer.

The EIG value is still `Σ P(r) × [H_before − H(B | simulated_answer_r)]`, averaged over the top-k recipes. The formula does not change; only the mechanics of what "applying the answer" means per branch.

### 4.3 `rank_questions()` and `should_ask()`

Both functions gain a `polar: bool = False` parameter that is threaded through to `expected_information_gain()`. Callers pass `polar=True` for the polar orchestrator and omit it (defaulting to `False`) for wh. No other logic changes.

---

## 5. `orchestrator_polar.py`

### 5.1 `parse_vlm_questions()`

The polar parser is identical to the wh parser except it also extracts the `proposition` field and drops any question missing a non-empty proposition. Without the proposition, neither the "yes" nor the "no" path can function — the incorporation target would be undefined.

### 5.2 `pick_human_answer_polar(question, bu) -> (answer, proposition)`

The human stub for the polar condition. Takes the full question dict (not just the question text) because it needs the proposition. Returns a tuple of `(answer, proposition)` where answer is `"yes"` or `"no"`.

Logic: embeds the proposition and computes its maximum cosine against the ground truth recipe's adaptive near-future window. If the cosine clears `HUMAN_STUB_MIN_COSINE` (0.40), the stub answers "yes" — the proposition matches something the cook is about to do. Otherwise "no" — the proposition doesn't fit the near future of the true recipe.

There is no "doesn't apply yet" branch. In the polar condition every question is answerable because a "no" is always a valid response, even to a proposition about a far-future or irrelevant step. The EIG gate handles propositions that are simply uninformative before they reach the stub.

In a real experiment this function is replaced with `input()` or audio capture with yes/no parsing. The stub is used only for automated testing.

### 5.3 Answer routing in the main loop

After the human answers, the orchestrator branches on the answer string:

```python
if answer == "yes":
    bu.incorporate_answer(proposition)
    snapshot(..., event_type="answer_yes", text=proposition, ...)
else:
    bu.incorporate_negative_answer(proposition)
    snapshot(..., event_type="answer_no", text=f"NO: {proposition}", ...)
```

The shadow updater (`bu_obs_only`) sees neither branch — it only receives video observations. This preserves the redundancy metric: the shadow updater represents what the system would know from observations alone, so the next window's IG difference correctly reflects how much unique value the question answer added.

The `pending_question` dict records `answer_type` as `"yes"` or `"no"` alongside the proposition text, so the question redundancy CSV captures the answer polarity for later analysis.

### 5.4 EIG scoring call

```python
ask, best_q, ranked = should_ask(
    belief_updater=bu,
    questions=questions,
    entropy_threshold=0.0,
    eig_threshold=EIG_THRESHOLD,
    questions_asked=questions_asked,
    max_questions=MAX_QUESTIONS,
    polar=True,
)
```

The only difference from the wh orchestrator's call is `polar=True`, which propagates through `rank_questions` and `expected_information_gain` to activate the yes/no simulation branches described in Section 4.

---

## 6. Information Flow Summary

### "Yes" path

```
VLM generates question with proposition
→ stub answers "yes"
→ bu.incorporate_answer(proposition)
    → proposition embedded, appended to observation sequence
    → _recompute() runs
        → hybrid F1 similarity computed over full obs sequence
        → negation penalties applied (from any prior "no" answers)
        → contrastive normalisation + softmax
    → IG_Q = H_before − H_after logged
```

### "No" path

```
VLM generates question with proposition
→ stub answers "no"
→ bu.incorporate_negative_answer(proposition)
    → proposition NOT appended to observation sequence
    → proposition and its embedding added to _negations / _negation_vecs
    → _recompute() runs
        → hybrid F1 similarity computed over full obs sequence (unchanged)
        → for each recipe: find max cosine(negation_vecs, unseen_scenes)
          if cosine >= NEGATION_MATCH_THRESHOLD:
              similarity[recipe] -= NEGATION_PENALTY × cosine
        → contrastive normalisation + softmax on penalised similarities
    → IG_Q = H_before − H_after logged (with "NO: " prefix on answer string)
```

The penalty survives all subsequent observations because `_recompute()` re-derives it from `_negation_vecs` on every call. The negation does not get wiped out when the next clip is observed.

---

## 7. Design Constraints Preserved

**Fixed temperature.** `NEGATION_PENALTY` and `NEGATION_MATCH_THRESHOLD` are session constants, mirroring the fixed-temperature constraint. If either value varied across steps, IG_Q from "no" answers at different points in the session would not be on the same scale.

**No Bayesian carry-forward.** The negation is stored as evidence (`_negation_vecs`), not as a modification to `self.belief`. Every recompute rebuilds the distribution from scratch, treating negations as input to the similarity computation rather than as post-hoc adjustments to the distribution. This is architecturally consistent with how observations are handled.

**Contrastive normalisation unaffected.** The penalty is subtracted before the contrastive mean, so the normalisation step continues to operate correctly. Recipes that are not implicated by any "no" answer receive no penalty and are unaffected by the shift in the mean caused by penalised competitors.

**EIG simulation correctness.** `copy()` propagates `_negations` and `_negation_vecs` to the clone, so simulated branches inside `expected_information_gain()` accurately predict what the posterior would look like after a "yes" or "no" — including the compounding effect of prior negations.

---

## 8. Parameters Reference

| Parameter | Value | Location | Purpose |
|---|---|---|---|
| `NEGATION_PENALTY` | 0.5 | `belief_updater_v3.py` | Strength of downward pull on similarity for matched recipes |
| `NEGATION_MATCH_THRESHOLD` | 0.5 | `belief_updater_v3.py` | Min cosine for a recipe's unseen scenes to be penalised |
| `HUMAN_STUB_MIN_COSINE` | 0.40 | `orchestrator_polar.py` | Min cosine for stub to answer "yes" to a proposition |
| `EIG_THRESHOLD` | 0.10 bits | `orchestrator_polar.py` | Min EIG for a question to be asked |
| `ENTROPY_THRESHOLD` | 1.5 bits | `orchestrator_polar.py` | Gate: skip questioning if entropy already below this |
| `TOP_PROB_CEILING` | 0.75 | `orchestrator_polar.py` | Gate: skip questioning if top recipe already above this |
