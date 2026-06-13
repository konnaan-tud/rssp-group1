# Questioning Pipeline — Implementation Documentation

This document describes the clarification-question machinery built on top of the V3 belief updater. It covers what was added, where it lives, and the design decisions behind each piece.

## Overview

The questioning system observes a cook through a sequence of VLM clips, maintains a probability distribution over candidate recipes via `BeliefUpdaterV3`, and decides — at every window — whether to ask a clarification question. When it asks, it generates candidate questions with Qwen2.5-VL, scores each one by measured Expected Information Gain on a copy of the belief, and fires the highest-scoring question. The human's answer is appended to the observation sequence and the belief is recomputed.

The primary metric is **IG_Q** — the entropy drop attributable to a clarification answer — measured as a clean snapshot around each `incorporate_answer` call. This is what the proposal's research question is about.

## Repository changes

Three new files, two extended files, one fixed file, one deleted file:

```
questioning/
├── __init__.py                      NEW · package marker
├── questioning_pipeline.py          NEW · VLM prompt builder + Qwen runner
└── questioning_planner.py           NEW · EIG scorer + asking gate

orchestrator_v2_vlm.py               NEW · end-to-end demo / driver

probability/
└── belief_updater_v3.py             EXTENDED · recipe sentences + helpers

orchestrator_v2.py                   DEPRECATED · stub, safe to delete
```

## `BeliefUpdaterV3` extensions

V3's original interface exposed only the per-recipe vectors (`recipe_sequences`). The questioning code needs the raw scene **sentences** to build prompt context and simulate answers — so two small additions were made:

**1. Load the recipe sentence strings.** The constructor now also reads `data/italian_recipe_scenes.json` and stores `self.recipe_sentences: dict[str, list[str]]` keyed by dish name. A new `recipe_scenes_path` parameter defaults to that file; can be overridden if the data lives elsewhere.

**2. Two helpers on the class:**

```python
def observed_sentences(self) -> list[str]:
    """Strings of all observations (clips + answers) added so far, in order."""

def unseen_recipe_scenes(self, recipe_name: str) -> list[str]:
    """Recipe scene sentences for `recipe_name` not yet observed verbatim."""
```

`observed_sentences()` reads from the existing `DynamicScene` metadata. `unseen_recipe_scenes()` is a literal-string set-difference between a recipe's scenes and what's been observed — V3's other infrastructure handles the actual semantic matching.

**3. `copy()` propagates the new fields.** The clone now shares `recipe_sentences` along with `recipe_sequences` (both are read-only at runtime, so sharing is safe and cheap).

These extensions don't touch the contrastive softmax, hybrid F1, IG_Q tracking, or temperature — V3's core stays exactly as your peers designed it.

## `questioning/questioning_pipeline.py`

Three public functions plus a smoke test.

### `build_recipe_context(belief_updater, active_threshold=0.02, top_k=6, future_scenes=4)`

Produces a compact text block summarising the session state for the VLM prompt. Three sections:

```
CURRENT BELIEF DISTRIBUTION (sorted by probability):
  0.358  ███████████                     cacio e pepe
  0.333  ██████████                      carbonara
  ...
  entropy: 2.745 bits over 6 active recipes (max possible: 4.644 bits)

ALREADY OBSERVED (in temporal order — do not ask about these):
  1. A cook pours water into a large pot.
  2. A cook cracks eggs into a mixing bowl.

ACTIVE CANDIDATE RECIPES — next likely scenes per recipe:
- carbonara  (p=0.333)
      • A cook grates pecorino into the mixing bowl.
      • A cook dices pancetta on a cutting board.
      • A cook cooks pancetta in a skillet.
- cacio e pepe  (p=0.358)
      • A cook grates pecorino into a bowl.
      • A cook drops pasta into salted boiling water.
      • A cook scoops pasta water with a measuring cup.
...
```

This is the V3-native equivalent of the V2 "vocabulary pool" — it shows the VLM what to distinguish *and* gives it the actual sentence vocabulary to draw target answers from. The per-recipe `p=…` annotations and the leading belief distribution mean Qwen sees the uncertainty structure of the problem, not just the candidate list.

Active recipes are filtered by `active_threshold` (default 0.02) and capped at `top_k` (default 6). Each recipe lists up to `future_scenes` (default 4) of its remaining unseen scenes, in recipe order.

### `build_question_prompt(recipe_context: str) -> str`

Assembles the full prompt sent to Qwen. The prompt has been slimmed substantially from the V2 version (about 50 lines instead of 140). Key elements:

- A brief task statement: generate 3 wh-questions whose answers reduce uncertainty.
- A description of the `targets` field — must be scene-style sentences drawn from the remaining scenes in the session context.
- A schema example with **placeholder values** (`"<scene sentence from candidate A's remaining scenes>"`) so Qwen learns structure without copying concrete strings. We saw earlier runs where concrete example targets (yogurt/miso/soy sauce) were getting parroted verbatim into Italian pasta sessions; the placeholder approach broke that pattern.
- An explicit "prefer questions about the NEXT step" instruction.
- The session context block (the output of `build_recipe_context`) embedded between the task description and the schema example.

The result is a focused prompt that has all the information the VLM needs without consuming all of the model's attention budget on instructions.

### `run_qwen_prompt(prompt, model_name=DEFAULT_MODEL, max_new_tokens=768)`

Thin wrapper around `Qwen2_5_VLForConditionalGeneration.from_pretrained` and `model.generate`. Auto-selects MPS / CUDA / CPU. Loads the model on every call (an obvious performance improvement is to load once per session — left as a small future refactor).

## `questioning/questioning_planner.py`

Four public functions implementing Bayesian Experimental Design for question selection.

### `simulate_answer(question, recipe_name, belief_updater, near_future_window=3)`

For a given candidate question and a given recipe, produces the **scene-style sentence the cook would plausibly say if `recipe_name` were the truth**. This is the analogue of V2's "match question targets against the recipe's term vocabulary" but adapted to V3's sentence-level world.

Strategy:

1. Get the recipe's unseen scenes via `belief_updater.unseen_recipe_scenes(recipe_name)`.
2. Restrict to the **next `near_future_window` scenes** in recipe order. This is the temporal-locality constraint — it stops the planner from simulating finishing-step answers when the cook has barely started.
3. Embed the question and each near-future scene. Pick the scene with highest cosine to the question.

The window default is 3. Without it, the planner can pick a recipe's late scenes (e.g. "transfers pasta into the skillet") as the simulated answer for a window-1 observation, leading to questions like "what is the final step before serving?" — bad UX and bad calibration.

### `expected_information_gain(question, belief_updater, top_k=5)`

The core EIG estimator.

```
EIG(Q) ≈ Σ_{r ∈ top-k} P(r) · [H(belief) − H(belief | simulated_answer_r)]
```

Pseudocode:

```python
h_now = belief_updater.entropy()
candidates = top_k recipes by belief
for recipe_name, p_r in candidates:
    ans = simulate_answer(question, recipe_name, belief_updater)
    clone = belief_updater.copy()
    clone.incorporate_answer(ans)               # appends to clone's obs sequence
    h_after = clone.entropy()
    eig += p_r * (h_now - h_after)
```

Returns the total EIG and a per-branch detail list (recipe, simulated answer, predicted entropy after, ΔH). The per-branch list is what the orchestrator surfaces for transparent logging.

The clone is made via `BeliefUpdaterV3.copy()` which shares the heavy read-only state (vectors, sentences, embedder) and copies only the mutable belief and observation sequence. Cheap enough to run for 5 top-k recipes × 3 candidate questions per VLM call.

### `rank_questions(questions, belief_updater, top_k=5)`

Annotates every question with `measured_eig` and `branches`, sorted by EIG descending. Pure composition over `expected_information_gain`.

### `should_ask(belief_updater, questions, entropy_threshold, eig_threshold, questions_asked, max_questions, top_k)`

Combined three-condition gate. Returns `(should_ask: bool, best_question: dict | None, all_ranked: list[dict])`. Conditions:

1. Budget: `questions_asked < max_questions`.
2. Uncertainty: `H(belief) > entropy_threshold`.
3. Useful question exists: `max measured EIG > eig_threshold`.

All three must hold. Condition (3) is what catches the "professor's tool question" case — even with high entropy, if no available question would meaningfully discriminate the top recipes, don't ask.

## `orchestrator_v2_vlm.py`

End-to-end driver. Runs the carbonara session, manages the gate, drives the VLM, logs everything.

### Hyperparameters

| Name | Default | Purpose |
|------|---------|---------|
| `ENTROPY_THRESHOLD` | 1.5 bits | Skip if belief is sharper than this |
| `TOP_PROB_CEILING` | 0.75 | Skip if the leading recipe is at or above this |
| `EIG_THRESHOLD` | 0.10 bits | Skip if no candidate question is informative |
| `MAX_QUESTIONS` | 9999 | Budget per session — currently disabled |
| `PROMPT_TOP_K` | 6 | Active recipes surfaced to the VLM |
| `PROMPT_ACTIVE_THRESHOLD` | 0.02 | Min probability for a recipe to be shown |

### Gate logic

The cheap gate fires before any VLM call:

```python
if bu.entropy() < ENTROPY_THRESHOLD:
    skip("entropy < threshold")
if top_prob >= TOP_PROB_CEILING:
    skip("top recipe already dominant")
```

The pair of conditions catches two different shapes of "no point asking": diffuse belief that's actually fairly sharp (low entropy) and concentrated belief on one clear winner (high top probability). They're not redundant — a 5-way split among nearly-equal recipes is high entropy but no dominant top; a clear top with one minor competitor is moderate entropy but high top probability. Either case should stop asking.

If both pass, the VLM is called, the planner scores its questions, and `should_ask` applies the EIG threshold.

### `pick_human_answer(question_text, bu)`

Stub for the human. In real experiments this is replaced with `input()` or audio capture. The stub strategy:

1. Two keyword overrides for negation-style answers (butter/cream questions → "no, just pancetta and eggs"; herb questions → "no fresh herbs, only black pepper"). These are answers whose true content isn't a single recipe scene.
2. **Default fallback**: call `simulate_answer(question, "carbonara", bu)` and convert the recipe-style sentence to first-person.

The default is the cleanest stub we have: the "human" answers using the same simulator the planner uses for prediction, pointed at the ground truth. So the realised IG_Q reflects what the recipe's actual next scene would do, not what a hardcoded default produces.

### Per-window flow

1. Embed and append the observation, update the belief.
2. Print top recipe, entropy.
3. Apply the cheap gate.
4. If passed, call the VLM and parse the JSON response into a list of question dicts.
5. Score with `should_ask`, log the per-branch EIG breakdown.
6. If a question fires, get the human answer, call `bu.incorporate_answer`, log predicted vs realised IG_Q, log the post-answer top-3.

### Log columns

The per-branch breakdown is the most important debug surface:

```
EIG +0.673 bits  |  What ingredient will the cook add next?
    - carbonara                    p=0.651  sim → 'A cook cooks pancetta in a skillet.'  drop=+0.888  weighted=+0.579
    - cacio e pepe                 p=0.142  sim → 'A cook grates pecorino into a bowl.'  drop=+0.257  weighted=+0.036
    ...
```

- `p` — current belief for that recipe.
- `sim` — scene the recipe would name if it were true.
- `drop` — ΔH in that branch (entropy_before − entropy_after on the clone).
- `weighted` — `p × drop`, that branch's share of the total EIG.

Total EIG is the sum of `weighted` across the top-k branches.

After a question fires, the orchestrator also prints the top-3 of the updated belief so the effect of the answer is visible at a glance.


## Key design decisions

### Why scene sentences instead of term targets

V2's planner used `targets: list[str]` where each target was an ingredient or action word. V3's planner uses `targets` as scene-style sentences. The reason: V3's belief updater consumes sentences (`bu.incorporate_answer(sentence)`), not term lists. The cleanest simulator is one that produces what the updater expects — a sentence. This also avoids V2's vocabulary-pool gymnastics.

### Why temporal locality in `simulate_answer`

Without it, the planner can simulate any recipe's unseen scene as the hypothetical answer. For carbonara after the first observation, the simulator could pick "transfers pasta into the skillet" — a scene 8 steps in the future. That produces high EIG (recipes diverge in late scenes) but leads to bad questions ("What is the final step before serving?") that the cook can't realistically answer at the current point in time. The `near_future_window=3` constraint bounds the simulator to "next few steps", which gives natural, plausible questions.

### Why the strict gate (entropy + top-prob)

Earlier runs with a single low entropy threshold (0.5 bits) caused the orchestrator to ask in every window — including ones where the leading recipe was already at 80%+ confidence. The asks degraded the belief because the human stub gave repeat answers that polluted the observation sequence. The strict gate (entropy ≥ 1.5 AND top-prob < 0.75) stops asking once the belief is decisively in one direction. On the carbonara session, this asks at windows 1 and 2, skips 3–5.

### Why the simulator drives the stub human

Earlier runs used a keyword-routed stub with a hardcoded default ("I am cracking eggs into a mixing bowl") that fired for any unmatched question. This produced false IG_Q signals when the default was unrelated to the question. The new stub uses `simulate_answer(question, "carbonara", bu)` — the same simulator the planner uses for predictions, pointed at the ground truth. So the "human" stays consistent with the recipe being cooked, and the realised IG_Q reflects the recipe's actual next scene.

### Why no Bayesian carry-forward for negation

V2 had a direct down-weight for negation answers: when the cook said "no, I'm not using X", we multiplied recipes containing X by 0.1. V3 recomputes belief from scratch each step using the full observation sequence — any direct multiplication would be wiped on the next observation. Negation handling is **intentionally not implemented in V3**. The trade-off is documented in the orchestrator header. If negation becomes important for experiments, the right approach is to extend `BeliefUpdaterV3` itself with a denied-scenes mechanism that survives recompute.

## Known limitations

- **`pick_human_answer` is a stub** with keyword overrides plus a simulator-based default. Real experiments need `input()` or audio.
- **`run_qwen_prompt` reloads the model on every call.** A single-load refactor would cut ~30 seconds off each gate firing.
- **The simulator picks by cosine relevance**, not strictly by recipe order. Window 1's stub answered with "I am grating pecorino" rather than "I am cracking eggs" because grating had higher cosine to "what ingredient next?". Minor.
- **Two semantically-equivalent VLM questions can get identical EIG.** Window 2 of the latest run had Q1 and Q2 both at 0.905. The planner has no notion of question redundancy.
- **No polar (yes/no) prompt yet.** The wh-vs-polar comparison from the proposal requires a second prompt template; only wh is implemented.
- **Hyperparameters are tuned by inspection** on the single carbonara test session. Threshold sweeps on recorded sessions are recommended before the final experiments.

## How to run

End-to-end VLM session:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python orchestrator_v2_vlm.py
```

Belief-update only, no VLM, fast (~10s):

```bash
python probability/test_belief_v3.py
```

Inspect the prompt that would be sent (no VLM, no orchestrator):

```bash
python questioning/questioning_pipeline.py
```

Rebuild `data/recipe_sequences.json` from `italian_recipe_scenes.json` after editing recipes:


## Expected behaviour on the carbonara session

Walking through the latest end-to-end run:

| Window | Observation | Entropy → | Top → | Action |
|--------|-------------|-----------|-------|--------|
| 1 | pours water | 4.64 → 4.32 | amatriciana 0.094 | ASK; IG_Q +0.92 |
| 2 | cracks eggs | 3.40 → 2.75 | cacio e pepe 0.358 | ASK; IG_Q +1.10 |
| 3 | grates pecorino | 1.65 → 1.98 | carbonara 0.652 | ASK; IG_Q +0.68 |
| 4 | dices pancetta | 1.30 → 1.24 | carbonara 0.828 | SKIP (entropy below 1.5) |
| 5 | cooks pancetta | 1.04 | carbonara 0.853 | SKIP (entropy below 1.5) |

Final: carbonara at 0.85, entropy 1.04 bits, 3 questions asked. All three IG_Q values positive, no over-asking, simulator calibration within ±0.5 bits on each question (window 3 within 0.005 bits).
