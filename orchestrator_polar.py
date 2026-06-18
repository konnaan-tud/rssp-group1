"""
orchestrator_polar.py
---------------------
End-to-end polar (yes/no) condition of the V3 pipeline.

Structurally identical to orchestrator_wh.py except:
  - The VLM is prompted for yes/no questions via build_question_prompt_polar.
    Each question carries a 'proposition' field (a scene-style statement) in
    addition to the question text.
  - The human stub answers "yes" or "no" based on whether the proposition
    matches carbonara's near-future scenes:
        yes  → proposition cosine >= HUMAN_STUB_MIN_COSINE against carbonara
        no   → proposition cosine <  HUMAN_STUB_MIN_COSINE (wrong for carbonara)
  - A "yes" is incorporated via bu.incorporate_answer(proposition) — the
    affirmed statement is treated as an observation.
  - A "no" is incorporated via bu.incorporate_negative_answer(proposition) —
    the denied statement is stored as persistent negative evidence that
    penalises recipes whose near-future matches it on every recompute.
  - should_ask / rank_questions are called with polar=True so EIG simulation
    uses simulate_polar_answer (yes/no branches) instead of simulate_answer.

Run:
    python orchestrator_polar.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from probability.belief_updater_v3 import BeliefUpdaterV3
from observation_pipeline.video_observer import load_qwen_vlm, describe_clip
from questioning.questioning_pipeline import (
    build_recipe_context,
    build_question_prompt_polar,
    generate_text_with_model,
)
from questioning.questioning_planner import should_ask
from utils.session_logger import SessionLogger
from utils.clip_loader import available_recipes, discover_clips
from dialogue.answer_normalizer import (
    AnswerOutcome,
    AnswerType,
    classify_polar_answer,
)


# ─────────────────────────────────────────────────────────────────────────
# Observation trajectory
# ─────────────────────────────────────────────────────────────────────────

# WINDOWS and GROUND_TRUTH are no longer hardcoded here — they are
# resolved at `main()` call time from `data/clips/<recipe>/` via
# `discover_clips`. Override with `--recipe pesto` (or any other folder
# name under data/clips/).

HUMAN_STUB_MIN_COSINE = 0.40


# ─────────────────────────────────────────────────────────────────────────
# Human-answer stub (polar)
# ─────────────────────────────────────────────────────────────────────────

def pick_human_answer_polar(
    question: dict,
    bu: BeliefUpdaterV3,
    ground_truth: str,
) -> tuple[str, str]:
    """
    Return (answer, proposition) where answer is "yes" or "no".

    Checks cosine between the proposition and the ground-truth recipe's
    adaptive near-future scenes. Above HUMAN_STUB_MIN_COSINE → "yes",
    else "no". In a real experiment this is replaced with input() or
    audio capture.
    """
    from questioning.questioning_planner import NEAR_FUTURE_WINDOW, _cosine, _adaptive_window

    proposition = (question.get("proposition") or "").strip()
    if not proposition:
        return "no", ""

    near = bu.unseen_recipe_scenes(ground_truth)
    window = _adaptive_window(bu, ground_truth, base=NEAR_FUTURE_WINDOW)
    near = near[:window]

    if not near:
        return "no", proposition

    p_vec      = bu._embedder.embed(proposition)
    scene_vecs = bu._embedder.embed_batch(near)
    best_cos   = max(_cosine(p_vec, v) for v in scene_vecs)

    answer = "yes" if best_cos >= HUMAN_STUB_MIN_COSINE else "no"
    return answer, proposition


def prompt_yesno_answer(
    question_text: str,
    proposition: str,
) -> tuple[str | None, str]:
    """
    Typed-text polar answer. Shows the question AND the proposition so the
    user knows exactly what they are confirming or denying — a "yes" here
    incorporates `proposition` as an observation, a "no" stores it as
    persistent negative evidence.

    Returns
    -------
    (answer, proposition)
        answer = "yes" | "no" | None ; None means "skip this question"
        proposition is echoed back unchanged so the caller can route it.
    """
    print(f"\n  >> Asking    : {question_text}")
    print(f"     proposition: {proposition}")
    print("     (type 'y' to confirm, 'n' to deny, or 'skip')")
    while True:
        try:
            raw = input("     your answer (y/n/skip) : ").strip().lower()
        except EOFError:
            return None, proposition
        if raw in ("y", "yes"):
            return "yes", proposition
        if raw in ("n", "no"):
            return "no", proposition
        if raw in ("", "skip", "s", "dunno", "don't know", "?"):
            return None, proposition
        print("     [input] please type 'y', 'n', or 'skip'.")


def get_polar_human_answer(
    question: dict,
    bu: BeliefUpdaterV3,
    answer_mode: str,
    ground_truth: str,
) -> tuple[str | None, str]:
    """Dispatch to the configured polar-answer source."""
    if answer_mode == "text":
        return prompt_yesno_answer(
            question["question"],
            question.get("proposition", ""),
        )
    return pick_human_answer_polar(question, bu, ground_truth)


# ─────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────

ENTROPY_THRESHOLD       = 1.5
TOP_PROB_CEILING        = 0.75
EIG_THRESHOLD           = 0.10
MAX_QUESTIONS           = 9_999
PROMPT_TOP_K            = 6
PROMPT_ACTIVE_THRESHOLD = 0.02


# ─────────────────────────────────────────────────────────────────────────
# VLM output parsing (polar)
# ─────────────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_vlm_questions(raw_response: str) -> list[dict]:
    """
    Parse Qwen's polar JSON output. Drops questions missing 'proposition'
    since both yes and no paths depend on it.
    """
    text  = _FENCE_RE.sub("", raw_response).strip()
    start = text.find("{")
    end   = text.rfind("}")
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
        proposition   = (q.get("proposition") or "").strip()
        if not question_text:
            continue
        if not proposition:
            print(f"  [parse] Question {i+1} missing 'proposition' — dropped.")
            continue
        category = q.get("category", "")
        if isinstance(category, str):
            category = category.strip().lower()
        if category not in ("ingredient", "method", "sequence"):
            category = "unspecified"
        distinguishes = q.get("distinguishes", [])
        clean.append({
            "rank":          q.get("rank", i + 1),
            "question":      question_text,
            "question_form": "polar",
            "category":      category,
            "proposition":   proposition,
            "distinguishes": distinguishes if isinstance(distinguishes, list) else [],
        })
    return clean


# ─────────────────────────────────────────────────────────────────────────
# VLM question generation
# ─────────────────────────────────────────────────────────────────────────

def generate_vlm_questions(
    belief_updater: BeliefUpdaterV3,
    model,
    processor,
    device: str,
) -> list[dict]:
    """Build the polar prompt, call Qwen via Ollama, parse JSON response."""
    context = build_recipe_context(
        belief_updater,
        active_threshold=PROMPT_ACTIVE_THRESHOLD,
        top_k=PROMPT_TOP_K,
    )
    n_active = sum(
        1 for p in belief_updater.belief.values()
        if p >= PROMPT_ACTIVE_THRESHOLD
    )
    print(f"  [VLM] Active recipes in context: {min(n_active, PROMPT_TOP_K)}")

    prompt = build_question_prompt_polar(recipe_context=context)
    print("  [VLM] Generating candidate polar questions...")
    raw_response = generate_text_with_model(prompt, model, processor, device)

    print("\n  [VLM raw response]")
    print("  " + raw_response.replace("\n", "\n  "))

    questions = parse_vlm_questions(raw_response)
    print(f"\n  [VLM] Parsed {len(questions)} candidate question(s).")
    return questions


# ─────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────

def main(answer_mode: str = "stub", recipe: str = "carbonara"):
    # 0. Resolve clips + ground-truth label for this dish.
    try:
        windows_list, ground_truth = discover_clips(recipe)
    except (FileNotFoundError, ValueError) as e:
        print(f"[clips] {e}", file=sys.stderr)
        return
    WINDOWS = windows_list
    GROUND_TRUTH = ground_truth
    print(f"[session] recipe={recipe!r}  ground_truth={GROUND_TRUTH!r}  "
          f"{len(WINDOWS)} clips")

    missing = [p for p in WINDOWS if not p.exists()]
    if missing:
        print("Missing clip files:")
        for p in missing:
            print(f"  - {p}")
        print("\nRun: python scripts/prepare_clips.py")
        return

    bu          = BeliefUpdaterV3()
    bu_obs_only = bu.copy()   # shadow updater — observations only, no answers

    print()
    model, processor, device = load_qwen_vlm()

    logger = SessionLogger(condition="polar", ground_truth=GROUND_TRUTH)

    history_rows: list[dict] = []

    def snapshot(step_num: int, event_type: str, text: str,
                 entropy_before: float) -> None:
        ent       = bu.entropy()
        top, top_p = bu.top_recipe()
        row = {
            "step":       step_num,
            "type":       event_type,
            "text":       text,
            "entropy":    round(ent, 4),
            "ig":         round(entropy_before - ent, 4),
            "top_recipe": top,
            "top_prob":   round(top_p, 4),
        }
        for name, prob in bu.belief.items():
            row[f"p::{name}"] = round(prob, 6)
        history_rows.append(row)

    snapshot(step_num=0, event_type="initial", text="",
             entropy_before=bu.entropy())

    question_log: list[dict]     = []
    pending_question: dict | None = None
    questions_asked               = 0

    print(f"\nMax entropy: {bu.max_entropy():.3f} bits")

    for i, clip_path in enumerate(WINDOWS, start=1):
        print(f"\n── Window {i}: {clip_path.name} ──────────────────────")

        # 1. Video observation.
        sentence = describe_clip(clip_path, model, processor, device)
        print(f"  observation : {sentence}")
        if not sentence:
            print("  [warn] VLM returned empty observation; skipping update.")
            continue

        # 2. Update both updaters.
        entropy_before_obs        = bu.entropy()
        shadow_entropy_before_obs = bu_obs_only.entropy()

        bu.update(sentence)
        bu_obs_only.update(sentence)

        real_ig_obs   = entropy_before_obs        - bu.entropy()
        shadow_ig_obs = shadow_entropy_before_obs - bu_obs_only.entropy()

        snapshot(step_num=i, event_type="observation",
                 text=sentence, entropy_before=entropy_before_obs)

        # Log observation.
        logger.log_observation(
            window=i, clip=clip_path.name, sentence=sentence,
            entropy_before=entropy_before_obs, entropy_after=bu.entropy(),
        )

        top, p = bu.top_recipe()
        print(f"  entropy : {bu.entropy():.4f} bits  (IG_obs {real_ig_obs:+.4f})")
        print(f"  top     : {top} ({p:.3f})")

        # 2b. Close out previous question redundancy.
        if pending_question is not None:
            redundancy   = shadow_ig_obs - real_ig_obs
            unique_value = pending_question["ig_q"] - redundancy
            pending_question.update({
                "next_obs_window":    i,
                "real_ig_next_obs":   round(real_ig_obs, 4),
                "shadow_ig_next_obs": round(shadow_ig_obs, 4),
                "redundancy":         round(redundancy, 4),
                "unique_value":       round(unique_value, 4),
            })
            print(f"  [redundancy] Q@W{pending_question['window']}: "
                  f"IG_Q={pending_question['ig_q']:+.3f}, "
                  f"next-obs redundancy={redundancy:+.3f}, "
                  f"unique value={unique_value:+.3f}")
            logger.log_redundancy(
                window=pending_question["window"],
                redundancy=redundancy,
                unique_value=unique_value,
            )
            question_log.append(pending_question)
            pending_question = None

        # 3. Cheap gate.
        if bu.entropy() < ENTROPY_THRESHOLD:
            print(f"  [gate] Entropy {bu.entropy():.3f} < {ENTROPY_THRESHOLD} — skipping.")
            continue
        if p >= TOP_PROB_CEILING:
            print(f"  [gate] Top recipe at {p:.3f} ≥ {TOP_PROB_CEILING} — skipping.")
            continue

        # 4. Generate polar candidate questions with the VLM.
        questions = generate_vlm_questions(bu, model, processor, device)
        if not questions:
            print("  [gate] No usable questions from VLM — skipping.")
            continue

        # 5. Score with the planner in polar mode.
        ask, best_q, ranked = should_ask(
            belief_updater=bu,
            questions=questions,
            entropy_threshold=0.0,
            eig_threshold=EIG_THRESHOLD,
            questions_asked=questions_asked,
            max_questions=MAX_QUESTIONS,
            polar=True,
        )

        # Per-branch EIG breakdown.
        print("\n  candidate questions (with EIG breakdown):")
        for q in ranked:
            cat  = q.get("category", "unspecified")
            prop = q.get("proposition", "")
            if len(prop) > 60:
                prop = prop[:57] + "..."
            print(f"    EIG {q['measured_eig']:+.3f} bits  |  [{cat}]  {q['question']}")
            print(f"         proposition: {prop}")
            for b in q["branches"]:
                if b["answer"] is None:
                    print(f"        - {b['recipe']:<28} p={b['p']:.3f}  (no simulated answer)")
                    continue
                weighted = b["p"] * b["delta_h"]
                print(f"        - {b['recipe']:<28} p={b['p']:.3f}  "
                      f"sim → {b['answer']!r:<6}  "
                      f"drop={b['delta_h']:+.3f}  weighted={weighted:+.3f}")

        if not ask:
            print("  [gate] Best EIG below threshold — skipping.")
            logger.log_skip()
            continue

        # 6. Ask and route the answer. Stub mode returns
        #    (answer="yes"|"no", proposition); text mode returns
        #    (answer="yes"|"no"|None, proposition) where None means the
        #    cook typed "skip" / "don't know" / "doesn't apply" / empty.
        answer, proposition = get_polar_human_answer(
            best_q, bu, answer_mode, GROUND_TRUTH,
        )

        if answer_mode != "text":
            print(f"\n  >> Asking    : {best_q['question']}")
            print(f"     proposition: {proposition}")
        print(f"     human      : {answer if answer is not None else '[skipped]'}")
        print(f"     predicted EIG : {best_q['measured_eig']:+.4f} bits")

        if not proposition:
            print("  [skip] No proposition — cannot incorporate, skipping.")
            continue

        if answer is None:
            # Typed-text path: cook said skip / don't know / doesn't apply.
            # No belief update, no IG_Q recorded — but snapshot AND log the
            # question so it shows on the entropy + redundancy plots.
            print("  [skip] Cook declined to answer — NO belief update, "
                  "no IG_Q recorded.")
            snapshot(
                step_num=i, event_type="answer_skipped",
                text=f"[skipped] {proposition}",
                entropy_before=bu.entropy(),
            )
            question_log.append({
                "window":             i,
                "question":           best_q["question"],
                "question_form":      "polar",
                "answer_type":        "skipped",
                "proposition":        proposition,
                "category":           best_q.get("category", "unspecified"),
                "predicted_eig":      round(best_q["measured_eig"], 4),
                "ig_q":               0.0,
                "next_obs_window":    None,
                "real_ig_next_obs":   None,
                "shadow_ig_next_obs": None,
                "redundancy":         None,
                "unique_value":       0.0,
            })
            logger.log_question(
                window=i, question=best_q["question"],
                category=best_q.get("category", ""),
                answer="", answer_type="skipped",
                predicted_eig=best_q["measured_eig"],
                realised_ig_q=0.0,
                proposition=proposition,
            )
            continue

        h_before = bu.entropy()

        if answer == "yes":
            bu.incorporate_answer(proposition)
            snapshot(step_num=i, event_type="answer_yes",
                     text=proposition, entropy_before=h_before)
        else:
            bu.incorporate_negative_answer(proposition)
            snapshot(step_num=i, event_type="answer_no",
                     text=f"NO: {proposition}", entropy_before=h_before)

        h_after  = bu.entropy()
        realised = h_before - h_after

        print(f"     realised IG_Q : {realised:+.4f} bits "
              f"(Δ vs predicted: {realised - best_q['measured_eig']:+.4f})")

        post_top = sorted(bu.belief.items(), key=lambda x: -x[1])[:3]
        print(f"     belief after Q:")
        for name, prob in post_top:
            bar = "█" * int(round(prob * 30))
            print(f"        {prob:.3f}  {bar:<30}  {name}")

        # Log question.
        logger.log_question(
            window=i,
            question=best_q["question"],
            category=best_q.get("category", ""),
            answer=proposition,
            answer_type=answer,       # "yes" or "no"
            predicted_eig=best_q["measured_eig"],
            realised_ig_q=realised,
            proposition=proposition,
        )

        pending_question = {
            "window":             i,
            "question":           best_q["question"],
            "question_form":      "polar",
            "answer_type":        answer,
            "proposition":        proposition,
            "category":           best_q.get("category", "unspecified"),
            "predicted_eig":      round(best_q["measured_eig"], 4),
            "ig_q":               round(realised, 4),
            "next_obs_window":    None,
            "real_ig_next_obs":   None,
            "shadow_ig_next_obs": None,
            "redundancy":         None,
            "unique_value":       round(realised, 4),
        }

        questions_asked += 1

    # ── Final report ──────────────────────────────────────────────────────
    top, p     = bu.top_recipe()
    is_correct = (top == GROUND_TRUTH)
    accuracy   = 1 if is_correct else 0

    print("\n" + "═" * 60)
    print(f"Ground truth     : {GROUND_TRUTH}")
    print(f"Final prediction : {top} (p={p:.4f})")
    print(f"Accuracy (Acc)   : {accuracy}  ({'CORRECT' if is_correct else 'WRONG'})")
    print(f"Final entropy    : {bu.entropy():.4f} bits")
    print(f"Questions asked  : {questions_asked}")
    print(f"IG_Q history     : {json.dumps(bu.ig_q_log(), indent=2)}")

    if pending_question is not None:
        pending_question["redundancy"]   = None
        pending_question["unique_value"] = pending_question["ig_q"]
        question_log.append(pending_question)

    # ── Session logger save ───────────────────────────────────────────────
    logger.save(
        predicted=top, accuracy=accuracy, final_prob=p,
        final_entropy=bu.entropy(), questions_asked=questions_asked,
        belief_history=history_rows, n_recipes=bu.N,
    )

    # ── Write CSVs ────────────────────────────────────────────────────────
    # Per-session CSVs go into the SessionLogger's session_dir; cross-session
    # session_summary.csv stays at the top-level outputs/ folder so it
    # accumulates across runs without overwriting.
    outputs_root = Path("outputs")
    outputs_root.mkdir(exist_ok=True)
    outputs_dir = logger.session_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    import csv as _csv
    if history_rows:
        # Same canonical filename across all three conditions so the plot
        # script can find it. The session folder already disambiguates by
        # condition (in the session_id).
        csv_path = outputs_dir / "belief_history.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)
        print(f"\nWrote belief history → {csv_path}")

    if question_log:
        q_csv = outputs_dir / "question_redundancy.csv"
        with open(q_csv, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=list(question_log[0].keys()))
            writer.writeheader()
            writer.writerows(question_log)
        print(f"Wrote question redundancy → {q_csv}")

    import datetime as _dt
    summary_csv = outputs_root / "session_summary.csv"
    summary_row = {
        "timestamp":          _dt.datetime.now().isoformat(timespec="seconds"),
        "condition":          "polar",
        "ground_truth":       GROUND_TRUTH,
        "predicted":          top,
        "accuracy":           accuracy,
        "predicted_prob":     round(p, 4),
        "final_entropy":      round(bu.entropy(), 4),
        "questions_asked":    questions_asked,
        "n_recipes":          bu.N,
        "n_windows":          len(WINDOWS),
        "total_ig_q":         round(
            sum(q.get("ig_q", 0) or 0 for q in bu.ig_q_log()), 4
        ),
        "total_unique_value": round(
            sum(q.get("unique_value", 0) or 0 for q in question_log), 4
        ),
    }
    summary_fields = list(summary_row.keys())
    write_header   = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=summary_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
    print(f"Appended session summary → {summary_csv}")

    print(f"\nPlot with: python scripts/plot_belief.py --session-dir {outputs_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the polar (yes/no) VLM orchestrator end-to-end."
    )
    parser.add_argument(
        "--answer-mode",
        choices=["stub", "text"],
        default="stub",
        help=(
            "Human-answer source. 'stub' uses the built-in "
            "cosine-against-proposition simulator; 'text' prompts for "
            "y/n/skip in the terminal."
        ),
    )
    recipes = available_recipes() or ["carbonara"]
    parser.add_argument(
        "--recipe",
        choices=recipes,
        default="carbonara" if "carbonara" in recipes else recipes[0],
        help=(
            "Dish to run. Clips live in data/clips/<recipe>/. "
            f"Detected recipes: {', '.join(recipes)}."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(answer_mode=args.answer_mode, recipe=args.recipe)