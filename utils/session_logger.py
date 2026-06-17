"""
utils/session_logger.py
-----------------------
Shared logging module used by all three orchestrators.

Per run, writes two outputs:

1. outputs/sessions/<session_id>.json
   Full detail: every observation sentence, IG_obs, every question asked,
   predicted EIG, realised IG_Q, redundancy, unique value, belief trajectory
   at question time. Self-contained record of one complete session.

2. outputs/runs_log.csv
   One row per session, all conditions. The comparison table for the paper.
   Append-only so it accumulates across all runs without overwriting.

Usage (same pattern in all three orchestrators):
    from utils.session_logger import SessionLogger

    logger = SessionLogger(condition="wh", ground_truth="carbonara")

    # after each observation
    logger.log_observation(
        window=i,
        clip=clip_path.name,
        sentence=sentence,
        entropy_before=entropy_before,
        entropy_after=bu.entropy(),
    )

    # after each question fires and is answered (wh / polar only)
    logger.log_question(
        window=i,
        question=best_q["question"],
        category=best_q.get("category", ""),
        answer=answer,
        answer_type="wh",          # or "yes" / "no" for polar
        predicted_eig=best_q["measured_eig"],
        realised_ig_q=realised,
        proposition=None,          # populated for polar
    )

    # when the next window's redundancy is closed out (wh / polar only)
    logger.log_redundancy(
        window=pending_question["window"],
        redundancy=redundancy,
        unique_value=unique_value,
    )

    # end of session
    logger.save(
        predicted=top,
        accuracy=accuracy,
        final_prob=p,
        final_entropy=bu.entropy(),
        questions_asked=questions_asked,
        belief_history=history_rows,   # the existing list of snapshot dicts
    )
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path


OUTPUTS_DIR  = Path("outputs")
SESSIONS_DIR = OUTPUTS_DIR / "sessions"
RUNS_LOG     = OUTPUTS_DIR / "runs_log.csv"

# Columns written to runs_log.csv — fixed order so all rows are aligned
# even across different conditions.
RUNS_LOG_FIELDS = [
    "session_id",
    "condition",
    "timestamp",
    "duration_s",
    "ground_truth",
    "predicted",
    "accuracy",
    "final_prob",
    "final_entropy",
    "questions_asked",
    "n_windows",
    "n_recipes",
    "total_ig_obs",
    "total_ig_q",
    "total_unique_value",
    "mean_ig_obs",
    "mean_ig_q",           # mean per question asked (None if 0 questions)
    "n_yes",               # polar only
    "n_no",                # polar only
    "n_question_skipped",  # windows where gate fired but EIG too low
    "notes",
]


class SessionLogger:
    """
    Accumulates per-window and per-question data during a session and
    flushes it to disk at the end.
    """

    def __init__(self, condition: str, ground_truth: str, notes: str = ""):
        """
        Parameters
        ----------
        condition    : "obs_only" | "wh" | "polar"
        ground_truth : recipe name string, e.g. "carbonara"
        notes        : optional free-text comment stored in JSON + CSV
        """
        self.condition    = condition
        self.ground_truth = ground_truth
        self.notes        = notes
        self.session_id   = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{condition}"
        )
        self._start       = time.time()

        # Per-window records — keyed by window number so updates from
        # log_redundancy can find and patch the right entry.
        self._windows: dict[int, dict] = {}
        self._question_skips: int = 0

        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def log_observation(
        self,
        window: int,
        clip: str,
        sentence: str,
        entropy_before: float,
        entropy_after: float,
    ) -> None:
        """Call once per window, right after bu.update(sentence)."""
        ig_obs = round(entropy_before - entropy_after, 4)
        self._windows[window] = {
            "window":          window,
            "clip":            clip,
            "observation":     sentence,
            "entropy_before":  round(entropy_before, 4),
            "entropy_after":   round(entropy_after, 4),
            "ig_obs":          ig_obs,
            "question":        None,   # filled in by log_question if fired
        }

    def log_question(
        self,
        window: int,
        question: str,
        category: str,
        answer: str,
        answer_type: str,        # "wh" | "yes" | "no"
        predicted_eig: float,
        realised_ig_q: float,
        proposition: str | None = None,
    ) -> None:
        """
        Call after the human answers and bu has been updated.
        answer_type should be:
          "wh"  for the wh condition (answer is a scene sentence)
          "yes" for a polar confirmation
          "no"  for a polar denial
        """
        entry = {
            "question":      question,
            "category":      category,
            "answer_type":   answer_type,
            "answer":        answer,
            "proposition":   proposition,
            "predicted_eig": round(predicted_eig, 4),
            "realised_ig_q": round(realised_ig_q, 4),
            "eig_error":     round(realised_ig_q - predicted_eig, 4),
            # redundancy fields filled in later by log_redundancy
            "redundancy":    None,
            "unique_value":  None,
        }
        if window in self._windows:
            self._windows[window]["question"] = entry
        else:
            # Window not yet logged (shouldn't happen, but be safe)
            self._windows[window] = {"window": window, "question": entry}

    def log_redundancy(
        self,
        window: int,
        redundancy: float,
        unique_value: float,
    ) -> None:
        """
        Call when the pending question's redundancy is closed out
        (i.e., when the NEXT window's observation arrives).
        Patches the question entry for `window`.
        """
        q = self._windows.get(window, {}).get("question")
        if q is not None:
            q["redundancy"]   = round(redundancy, 4)
            q["unique_value"] = round(unique_value, 4)

    def log_skip(self) -> None:
        """
        Call when the gate fires (entropy/top_prob check passed) but the
        best question's EIG was below threshold — i.e. a window where the
        system considered asking but decided not to.
        """
        self._question_skips += 1

    def save(
        self,
        predicted: str,
        accuracy: int,
        final_prob: float,
        final_entropy: float,
        questions_asked: int,
        belief_history: list[dict],
        n_recipes: int = 0,
    ) -> Path:
        """
        Flush everything to disk.

        Returns the path of the JSON session file.
        """
        duration = round(time.time() - self._start, 1)
        windows  = [self._windows[k] for k in sorted(self._windows)]

        # Aggregate stats
        ig_obs_values = [w["ig_obs"] for w in windows if w.get("ig_obs") is not None]
        total_ig_obs  = round(sum(ig_obs_values), 4)
        mean_ig_obs   = round(total_ig_obs / len(ig_obs_values), 4) if ig_obs_values else 0.0

        questions     = [w["question"] for w in windows if w.get("question")]
        total_ig_q    = round(sum(q["realised_ig_q"] for q in questions), 4)
        mean_ig_q     = (
            round(total_ig_q / len(questions), 4) if questions else None
        )
        total_unique  = round(
            sum(q["unique_value"] for q in questions if q.get("unique_value") is not None),
            4,
        )
        n_yes = sum(1 for q in questions if q.get("answer_type") == "yes")
        n_no  = sum(1 for q in questions if q.get("answer_type") == "no")

        # ── JSON session file ─────────────────────────────────────────────
        session = {
            "session_id":   self.session_id,
            "condition":    self.condition,
            "ground_truth": self.ground_truth,
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "duration_s":   duration,
            "notes":        self.notes,
            "result": {
                "predicted":       predicted,
                "accuracy":        accuracy,
                "final_prob":      round(final_prob, 4),
                "final_entropy":   round(final_entropy, 4),
                "questions_asked": questions_asked,
                "total_ig_obs":    total_ig_obs,
                "total_ig_q":      total_ig_q,
                "total_unique_value": total_unique,
                "mean_ig_obs":     mean_ig_obs,
                "mean_ig_q":       mean_ig_q,
                "n_yes":           n_yes,
                "n_no":            n_no,
                "n_question_skipped": self._question_skips,
            },
            "windows":        windows,
            "belief_history": belief_history,
        }

        json_path = SESSIONS_DIR / f"{self.session_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        print(f"[log] Session JSON → {json_path}")

        # ── runs_log.csv (append) ─────────────────────────────────────────
        row = {
            "session_id":          self.session_id,
            "condition":           self.condition,
            "timestamp":           session["timestamp"],
            "duration_s":          duration,
            "ground_truth":        self.ground_truth,
            "predicted":           predicted,
            "accuracy":            accuracy,
            "final_prob":          round(final_prob, 4),
            "final_entropy":       round(final_entropy, 4),
            "questions_asked":     questions_asked,
            "n_windows":           len(windows),
            "n_recipes":           n_recipes,
            "total_ig_obs":        total_ig_obs,
            "total_ig_q":          total_ig_q,
            "total_unique_value":  total_unique,
            "mean_ig_obs":         mean_ig_obs,
            "mean_ig_q":           mean_ig_q if mean_ig_q is not None else "",
            "n_yes":               n_yes,
            "n_no":                n_no,
            "n_question_skipped":  self._question_skips,
            "notes":               self.notes,
        }
        write_header = not RUNS_LOG.exists()
        with open(RUNS_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RUNS_LOG_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"[log] Appended to runs log → {RUNS_LOG}")

        return json_path