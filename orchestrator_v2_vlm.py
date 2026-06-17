"""
orchestrator_v2_vlm.py
----------------------
End-to-end demo of the V3 pipeline with VLM-generated clarification
questions.

What it does:
  1. Walks through a simulated carbonara session (one scene sentence per
     window) and feeds each clip to BeliefUpdaterV3.
  2. After every window the cheap gate (entropy threshold) decides whether
     it's worth calling the VLM at all.
  3. If yes, generates candidate clarification questions with Qwen2.5-VL,
     scores each by expected information gain on a *copy* of the belief,
     and applies the EIG gate.
  4. If a question fires, the hard-coded human (pick_human_answer) gives a
     scene-style answer that gets appended to the observation sequence via
     bu.incorporate_answer(...).
  5. Predicted EIG vs realised IG_Q is logged per question.

Negation handling is intentionally NOT implemented here. In V3 the belief
is recomputed from scratch each step, so the V2-style direct down-weight
would be wiped out on the next observation. If the embedder mishandles
negation answers (likely for the "I am not using X" case), that's a known
limitation noted in the docs.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python orchestrator_v2_vlm.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import os
from pathlib import Path

# Make package imports work when run from the repo root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from probability.belief_updater_v3 import BeliefUpdaterV3
from observation_pipeline.video_observer import load_qwen_vlm, describe_clip
from questioning.questioning_pipeline import (
    build_recipe_context,
    build_question_prompt,
    generate_text_with_model,
)
from questioning.questioning_planner import should_ask, simulate_answer
from dialogue.answer_normalizer import (
    AnswerOutcome,
    AnswerType,
    normalize_answer,
)
from utils.session_logger import SessionLogger

# ─────────────────────────────────────────────────────────────────────────
# Observation trajectory — carbonara session (REAL VIDEO).
# Each entry is a path to one cooking clip; the orchestrator runs each clip
# through Qwen2.5-VL to extract a scene sentence, then feeds that sentence
# to bu.update(). Build the clips with:
#     python scripts/prepare_clips.py
# ─────────────────────────────────────────────────────────────────────────

CLIPS_DIR = Path("data/clips")

WINDOWS = [
    CLIPS_DIR / "01_pour_water.mp4",
    CLIPS_DIR / "02_crack_egg.mp4",
    CLIPS_DIR / "03_grating_pecorino.mp4",
    CLIPS_DIR / "04_chopping_pancetta.mp4",
    CLIPS_DIR / "05_cooking_pancetta.mp4",
    CLIPS_DIR / "06_boil_pasta.mp4",
    CLIPS_DIR / "07_drain_pasta.mp4",
    CLIPS_DIR / "08_pasta_into_skillet.mp4",
    CLIPS_DIR / "09_stirring_carbonara.mp4",
]


# ─────────────────────────────────────────────────────────────────────────
# Human-answer stub
#
# The cook is making carbonara. pick_human_answer returns what the cook
# would plausibly say for the given question:
#   - Keyword overrides for questions whose natural answer is a negation
#     or summary that isn't a single recipe scene
#     (e.g. "Are you using butter?" → "no, just pancetta and eggs").
#   - Default: ask the planner's own simulator what scene the ground-truth
#     recipe (carbonara) would name for this question. That is the same
#     mechanism the planner uses to predict per-recipe answers, applied
#     to the actual recipe being cooked. This is a much better demo stub
#     than a hardcoded default — the human stays "in character" with the
#     recipe they're cooking.
#
# In real experiments this whole function is replaced with input() (typed
# human) or audio capture.
# ─────────────────────────────────────────────────────────────────────────

GROUND_TRUTH = "carbonara"

# Minimum cosine between the question and the simulated scene for the human
# stub to actually answer with that scene. If below this, the cook says
# "doesn't apply yet" — preventing false IG_Q signals from questions whose
# answer the cook can't really give.
HUMAN_STUB_MIN_COSINE = 0.40
HUMAN_STUB_NOT_AT_STEP = (
    "That doesn't apply yet — I'm not at that step of the recipe."
)


def pick_human_answer(question_text: str, bu: BeliefUpdaterV3) -> str:
    """
    Stub "human" for the carbonara session. Returns the raw string the
    cook would say; downstream normalisation runs through `normalize_answer`
    so this path looks identical to the typed-text path.

      - Keyword overrides for questions whose natural answer is a
        negation or summary, not a single recipe scene.
      - Default: pick the carbonara unseen scene most similar to the
        question (within the near-future window). If no scene is
        sufficiently similar, return the "doesn't apply yet" string —
        the normaliser will classify it as DOESNT_APPLY.

    In real experiments this whole function is replaced with the
    typed-text path (`--answer-mode text`) or with audio capture.
    """
    q = question_text.lower()

    # Overrides: questions whose true answer is a negation/summary, not a
    # single recipe scene.
    if any(w in q for w in ["butter", "cream"]) and any(w in q for w in ["fat", "base", "using"]):
        return ("No butter and no cream; the fat in carbonara comes from "
                "rendered pancetta and egg yolks.")
    if "herb" in q:
        return "Carbonara does not use fresh herbs, only black pepper."

    # Default: simulate against carbonara and check whether the picked scene
    # is actually relevant to the question.
    from questioning.questioning_planner import NEAR_FUTURE_WINDOW, _cosine

    near = bu.unseen_recipe_scenes(GROUND_TRUTH)[:NEAR_FUTURE_WINDOW]
    if not near:
        return HUMAN_STUB_NOT_AT_STEP

    q_vec = bu._embedder.embed(question_text)
    scene_vecs = bu._embedder.embed_batch(near)
    cosines = [_cosine(q_vec, v) for v in scene_vecs]
    best_idx = int(max(range(len(cosines)), key=lambda i: cosines[i]))
    best_cos = cosines[best_idx]

    if best_cos < HUMAN_STUB_MIN_COSINE:
        return HUMAN_STUB_NOT_AT_STEP

    # Return the recipe scene as-is. The normaliser will pass it through
    # unchanged (already in recipe form).
    return near[best_idx]


def prompt_text_answer(question_text: str) -> str:
    """
    Prompt a human for a typed answer in the terminal.

    Returns the raw text (possibly empty / "skip" / "no" / etc.); the
    normaliser figures out what to do with it. We deliberately do NOT
    short-circuit on "skip" here — the normaliser sees it as DONT_KNOW
    and the orchestrator's standard skip path handles it uniformly.
    """
    print(f"\n  >> Asking : {question_text}")
    print("     (type your answer, or 'skip' / 'don't know' / 'doesn't apply')")
    try:
        return input("     your answer : ").strip()
    except EOFError:
        return ""


def get_raw_human_answer(
    question_text: str,
    bu: BeliefUpdaterV3,
    answer_mode: str,
) -> str:
    """Dispatch to the configured human-answer source."""
    if answer_mode == "text":
        return prompt_text_answer(question_text)
    return pick_human_answer(question_text, bu)


# ─────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────

# Gate thresholds. Either condition being satisfied skips the question:
#   - entropy already sharp (low residual uncertainty)
#   - top recipe already dominant (one clear leader)
# Picked so the carbonara session asks at windows 1 and 2 (uncertain) but
# stops by window 3 once carbonara is at 80%+.
ENTROPY_THRESHOLD = 1.5      # bits; below this skip
TOP_PROB_CEILING = 0.75      # if top recipe at or above this, skip
EIG_THRESHOLD = 0.10         # bits; below this no question is worth asking

# Question budget — disabled for now (kept for easy re-enable).
# MAX_QUESTIONS = 2
MAX_QUESTIONS = 9_999

# Context prompt limits.
PROMPT_TOP_K = 6
PROMPT_ACTIVE_THRESHOLD = 0.02


# ─────────────────────────────────────────────────────────────────────────
# VLM output parsing
# ─────────────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_vlm_questions(raw_response: str) -> list[dict]:
    """
    Parse Qwen's JSON output into the dict shape that the planner expects.

    Tolerates code fences, leading commentary, missing fields, wrong types.
    Malformed entries are silently dropped.
    """
    text = _FENCE_RE.sub("", raw_response).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        print("  [parse] No JSON object found in VLM response.")
        return []

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        print(f"  [parse] JSON decode failed: {e}")
        return []

    raw_questions = parsed.get("questions", [])
    if not isinstance(raw_questions, list):
        print("  [parse] 'questions' field is missing or not a list.")
        return []

    clean: list[dict] = []
    for i, q in enumerate(raw_questions):
        if not isinstance(q, dict):
            continue
        question_text = (q.get("question") or "").strip()
        if not question_text:
            continue
        targets = q.get("targets", [])
        distinguishes = q.get("distinguishes", [])
        category = q.get("category", "")
        if isinstance(category, str):
            category = category.strip().lower()
        if category not in ("ingredient", "method", "sequence"):
            category = "unspecified"
        clean.append({
            "rank":          q.get("rank", i + 1),
            "question":      question_text,
            "question_form": q.get("question_form", "wh"),
            "category":      category,
            "targets":       targets if isinstance(targets, list) else [],
            "distinguishes": distinguishes if isinstance(distinguishes, list) else [],
        })
    return clean


def generate_vlm_questions(
    belief_updater: BeliefUpdaterV3,
    model,
    processor,
    device: str,
) -> list[dict]:
    """
    Build the prompt, call Qwen on the already-loaded model, parse the
    JSON response. Reuses the model loaded at session start by
    load_qwen_vlm() — no per-call reload.
    """
    context = build_recipe_context(
        belief_updater,
        active_threshold=PROMPT_ACTIVE_THRESHOLD,
        top_k=PROMPT_TOP_K,
    )
    n_active = sum(1 for p in belief_updater.belief.values()
                   if p >= PROMPT_ACTIVE_THRESHOLD)
    print(f"  [VLM] Active recipes in context: {min(n_active, PROMPT_TOP_K)}")

    prompt = build_question_prompt(recipe_context=context)
    print("  [VLM] Generating candidate questions...")
    raw_response = generate_text_with_model(prompt, model, processor, device)

    print("\n  [VLM raw response]")
    print("  " + raw_response.replace("\n", "\n  "))

    questions = parse_vlm_questions(raw_response)
    print(f"\n  [VLM] Parsed {len(questions)} candidate question(s).")
    return questions


# ─────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────

def main(answer_mode: str = "stub"):
    # 1. Sanity check: every clip must exist before we waste 30s on model load.
    missing = [p for p in WINDOWS if not p.exists()]
    if missing:
        print("Missing clip files:")
        for p in missing:
            print(f"  - {p}")
        print("\nRun: python scripts/prepare_clips.py")
        return

    bu = BeliefUpdaterV3()

    # Parallel shadow belief updater: sees observations only, never sees
    # clarification answers. Used to measure per-question redundancy —
    # specifically, how much of the NEXT observation's information was
    # "stolen" by the question.
    #
    # bu.copy() shares the embedder + recipe vectors (read-only) and starts
    # with the same (empty) observation sequence and uniform belief.
    bu_obs_only = bu.copy()

    # 2. Load Qwen2.5-VL once. Reused for BOTH:
    #    - observation (video → scene sentence)
    #    - question generation (text prompt → JSON)
    print()
    model, processor, device = load_qwen_vlm()
    logger = SessionLogger(condition="wh", ground_truth=GROUND_TRUTH)

    # 3. Belief-history tracker. One row per event (initial / observation /
    # answer). Saved to outputs/belief_history.csv at the end for plotting.
    history_rows: list[dict] = []

    def snapshot(step_num: int, event_type: str, text: str,
                 entropy_before: float) -> None:
        ent = bu.entropy()
        top, top_p = bu.top_recipe()
        row = {
            "step": step_num,
            "type": event_type,
            "text": text,
            "entropy": round(ent, 4),
            "ig": round(entropy_before - ent, 4),
            "top_recipe": top,
            "top_prob": round(top_p, 4),
        }
        for name, prob in bu.belief.items():
            row[f"p::{name}"] = round(prob, 6)
        history_rows.append(row)

    # Step 0 — initial uniform prior
    snapshot(step_num=0, event_type="initial", text="",
             entropy_before=bu.entropy())

    # Per-question redundancy log. Each entry is closed out at the NEXT
    # window's observation (by computing the gap between real and shadow
    # IG_obs). Saved to outputs/question_redundancy.csv at the end.
    question_log: list[dict] = []
    pending_question: dict | None = None

    questions_asked = 0
    print(f"\nMax entropy: {bu.max_entropy():.3f} bits")

    for i, clip_path in enumerate(WINDOWS, start=1):
        print(f"\n── Window {i}: {clip_path.name} ──────────────────────")

        # 1. Observe the clip — Qwen video → one scene sentence.
        sentence = describe_clip(clip_path, model, processor, device)
        print(f"  observation : {sentence}")
        if not sentence:
            print("  [warn] VLM returned empty observation; skipping update.")
            continue

        # 2. Observation update — apply to BOTH the real and shadow updaters.
        entropy_before_obs = bu.entropy()
        shadow_entropy_before_obs = bu_obs_only.entropy()

        bu.update(sentence)
        bu_obs_only.update(sentence)

        real_ig_obs = entropy_before_obs - bu.entropy()
        shadow_ig_obs = shadow_entropy_before_obs - bu_obs_only.entropy()

        snapshot(step_num=i, event_type="observation",
                 text=sentence, entropy_before=entropy_before_obs)
        logger.log_observation(
            window=i, clip=clip_path.name, sentence=sentence,
            entropy_before=entropy_before_obs, entropy_after=bu.entropy(),
        )

        top, p = bu.top_recipe()
        print(f"  entropy : {bu.entropy():.4f} bits  (IG_obs {real_ig_obs:+.4f})")
        print(f"  top     : {top} ({p:.3f})")

        # 2b. If a question fired in the PREVIOUS window, close out its
        # redundancy now. Redundancy = how much information the next
        # observation would have delivered WITHOUT the question, minus how
        # much it actually delivered WITH the question already incorporated.
        # Unique question contribution = IG_Q − redundancy.
        if pending_question is not None:
            redundancy = shadow_ig_obs - real_ig_obs
            unique_value = pending_question["ig_q"] - redundancy
            pending_question.update({
                "next_obs_window": i,
                "real_ig_next_obs": round(real_ig_obs, 4),
                "shadow_ig_next_obs": round(shadow_ig_obs, 4),
                "redundancy": round(redundancy, 4),
                "unique_value": round(unique_value, 4),
            })
            logger.log_redundancy(
                window=pending_question["window"],
                redundancy=redundancy,
                unique_value=unique_value,
            )
            print(f"  [redundancy] Q@W{pending_question['window']}: "
                  f"IG_Q={pending_question['ig_q']:+.3f}, "
                  f"next-obs redundancy={redundancy:+.3f}, "
                  f"unique value={unique_value:+.3f}")
            question_log.append(pending_question)
            pending_question = None

        # 2. CHEAP gate first — skip the VLM call when the answer is already
        # obvious. Budget gate is disabled for now (uncomment to restore).
        # if questions_asked >= MAX_QUESTIONS:
        #     print("  [gate] Question budget exhausted — skipping.")
        #     continue
        if bu.entropy() < ENTROPY_THRESHOLD:
            print(f"  [gate] Entropy {bu.entropy():.3f} < {ENTROPY_THRESHOLD} — skipping.")
            continue
        if p >= TOP_PROB_CEILING:
            print(f"  [gate] Top recipe at {p:.3f} ≥ {TOP_PROB_CEILING} — skipping.")
            continue

        # 3. EXPENSIVE step — generate candidate questions with the VLM.
        # Uses the same loaded model as the observation step.
        questions = generate_vlm_questions(bu, model, processor, device)
        if not questions:
            print("  [gate] No usable questions from VLM — skipping.")
            continue

        # 4. Score with the planner. entropy_threshold=0.0 here because we
        # already vetted entropy above; only the EIG and budget checks remain.
        ask, best_q, ranked = should_ask(
            belief_updater=bu,
            questions=questions,
            entropy_threshold=0.0,
            eig_threshold=EIG_THRESHOLD,
            questions_asked=questions_asked,
            max_questions=MAX_QUESTIONS,
        )

        # Per-branch EIG breakdown — for each candidate question, show what
        # each top-k recipe is hypothesised to answer and how much entropy
        # would drop in that branch.
        #   p          : current belief probability for that recipe
        #   sim answer : scene the recipe would name if it were the truth
        #   drop       : entropy drop in that branch (H_before − H_after)
        #   weighted   : p × drop, the branch's share of the total EIG
        # Total EIG = sum of weighted contributions across the top-k branches.
        print("\n  candidate questions (with EIG breakdown):")
        for q in ranked:
            cat = q.get("category", "unspecified")
            print(f"    EIG {q['measured_eig']:+.3f} bits  |  [{cat}]  {q['question']}")
            for b in q["branches"]:
                if b["answer"] is None:
                    print(f"        - {b['recipe']:<28} p={b['p']:.3f}  "
                          f"(no simulated answer)")
                    continue
                weighted = b["p"] * b["delta_h"]
                ans = b["answer"]
                if len(ans) > 50:
                    ans = ans[:47] + "..."
                print(f"        - {b['recipe']:<28} p={b['p']:.3f}  "
                      f"sim → {ans!r:<55}  "
                      f"drop={b['delta_h']:+.3f}  weighted={weighted:+.3f}")

        if not ask:
            print("  [gate] Best EIG below threshold — skipping.")
            logger.log_skip()
            continue

        # 5. Fire the chosen question. The stub path simulates a
        # carbonara cook; the text path prompts the terminal. Both
        # produce a raw string that the normaliser classifies.
        raw_answer = get_raw_human_answer(
            best_q["question"], bu, answer_mode=answer_mode,
        )
        outcome: AnswerOutcome = normalize_answer(raw_answer, belief_updater=bu)

        if answer_mode != "text":
            print(f"\n  >> Asking : {best_q['question']}")
        print(f"     raw answer    : {raw_answer!r}")
        print(f"     classified    : {outcome.type.value}")
        if outcome.recipe_sentence and outcome.type == AnswerType.AFFIRMATIVE:
            print(f"     recipe form   : {outcome.recipe_sentence}")
        print(f"     predicted EIG : {best_q['measured_eig']:+.4f} bits")

        # Route on the outcome's type. Three branches:
        #   - DOESNT_APPLY / DONT_KNOW / OFF_TOPIC → skip silently
        #   - NEGATIVE                              → log + skip (V3 can't
        #                                              represent negation
        #                                              reliably; documented
        #                                              limitation)
        #   - AFFIRMATIVE                           → incorporate + log IG_Q
        if outcome.skip_update:
            print(f"     [skip] {outcome.type.value} — NO belief update, "
                  f"no IG_Q recorded.")
            continue

        if outcome.type == AnswerType.NEGATIVE:
            negated = outcome.metadata.get("negated_entity")
            note = f" (negated entity: {negated!r})" if negated else ""
            print(f"     [skip] NEGATIVE answer{note} — NO belief update. "
                  f"V3's embedder does not represent negation reliably; "
                  f"recording as a missed-opportunity question.")
            # Still log the question so the redundancy CSV has a row
            # describing why no IG_Q was recorded.
            question_log.append({
                "window": i,
                "question": best_q["question"],
                "category": best_q.get("category", "unspecified"),
                "answer": raw_answer,
                "answer_type": outcome.type.value,
                "predicted_eig": round(best_q["measured_eig"], 4),
                "ig_q": 0.0,
                "next_obs_window": None,
                "real_ig_next_obs": None,
                "shadow_ig_next_obs": None,
                "redundancy": None,
                "unique_value": 0.0,
                "negated_entity": negated or "",
            })
            continue

        # AFFIRMATIVE path — actually update the belief.
        assert outcome.recipe_sentence is not None
        answer_sentence = outcome.recipe_sentence

        h_before = bu.entropy()
        bu.incorporate_answer(answer_sentence)
        # NOTE: do NOT apply the answer to bu_obs_only — that's the whole
        # point of the shadow updater. It only ever sees observations.
        snapshot(step_num=i, event_type="answer",
                 text=answer_sentence, entropy_before=h_before)
        h_after = bu.entropy()
        realised = h_before - h_after
        logger.log_question(
            window=i, question=best_q["question"],
            category=best_q.get("category", ""),
            answer=answer, answer_type="wh",
            predicted_eig=best_q["measured_eig"],
            realised_ig_q=realised,
        )

        print(f"     realised IG_Q : {realised:+.4f} bits "
              f"(Δ vs predicted: {realised - best_q['measured_eig']:+.4f})")

        # NEW: show the belief distribution after the answer so it's clear
        # how the question reshaped uncertainty.
        post_top = sorted(bu.belief.items(), key=lambda x: -x[1])[:3]
        print(f"     belief after Q:")
        for name, prob in post_top:
            bar = "█" * int(round(prob * 30))
            print(f"        {prob:.3f}  {bar:<30}  {name}")

        # Stash for next-window redundancy closure.
        pending_question = {
            "window": i,
            "question": best_q["question"],
            "category": best_q.get("category", "unspecified"),
            "answer": answer_sentence,
            "answer_type": outcome.type.value,
            "predicted_eig": round(best_q["measured_eig"], 4),
            "ig_q": round(realised, 4),
            "next_obs_window": None,
            "real_ig_next_obs": None,
            "shadow_ig_next_obs": None,
            "redundancy": None,
            "unique_value": round(realised, 4),  # placeholder; overwritten on close
            "negated_entity": "",
        }

        questions_asked += 1

    # Final report
    top, p = bu.top_recipe()
    is_correct = (top == GROUND_TRUTH)
    accuracy = 1 if is_correct else 0
    print("\n" + "═" * 60)
    print(f"Ground truth     : {GROUND_TRUTH}")
    print(f"Final prediction : {top} (p={p:.4f})")
    print(f"Accuracy (Acc)   : {accuracy}  ({'CORRECT' if is_correct else 'WRONG'})")
    print(f"Final entropy    : {bu.entropy():.4f} bits")
    print(f"Questions asked  : {questions_asked}")
    print(f"IG_Q history     : {json.dumps(bu.ig_q_log(), indent=2)}")

    # If the final question never got a follow-up observation, close it out
    # with redundancy unmeasured.
    if pending_question is not None:
        pending_question["redundancy"] = None
        pending_question["unique_value"] = pending_question["ig_q"]
        question_log.append(pending_question)

    # Write per-step belief history + per-question redundancy log to CSVs.
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    import csv as _csv
    if history_rows:
        csv_path = outputs_dir / "belief_history.csv"
        fieldnames = list(history_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history_rows)
        print(f"\nWrote belief history → {csv_path}")

    if question_log:
        q_csv = outputs_dir / "question_redundancy.csv"
        fieldnames = list(question_log[0].keys())
        with open(q_csv, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(question_log)
        print(f"Wrote question redundancy → {q_csv}")

    # Append a one-row summary to session_summary.csv. Lets the supervisor
    # aggregate accuracy across many sessions for the wh-vs-polar experiment.
    import datetime as _dt
    summary_csv = outputs_dir / "session_summary.csv"
    summary_row = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "ground_truth": GROUND_TRUTH,
        "predicted": top,
        "accuracy": accuracy,
        "predicted_prob": round(p, 4),
        "final_entropy": round(bu.entropy(), 4),
        "questions_asked": questions_asked,
        "n_recipes": bu.N,
        "n_windows": len(WINDOWS),
        "total_ig_q": round(
            sum(q.get("ig_q", 0) or 0 for q in bu.ig_q_log()), 4
        ),
        "total_unique_value": round(
            sum(q.get("unique_value", 0) or 0 for q in question_log), 4
        ),
    }
    summary_fields = list(summary_row.keys())
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=summary_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
    print(f"Appended session summary → {summary_csv}")

    print("\nPlot with: python scripts/plot_belief.py")
    logger.save(
        predicted=top, accuracy=accuracy, final_prob=p,
        final_entropy=bu.entropy(), questions_asked=questions_asked,
        belief_history=history_rows, n_recipes=bu.N,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the VLM orchestrator end-to-end."
    )
    parser.add_argument(
        "--answer-mode",
        choices=["stub", "text"],
        default="stub",
        help=(
            "Human-answer source. 'stub' uses the built-in carbonara "
            "simulator; 'text' prompts for a typed answer in the terminal."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(answer_mode=args.answer_mode)
