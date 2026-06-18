"""
scripts/plot_belief.py
----------------------
Read outputs/belief_history.csv (produced by orchestrator_v2_vlm.py) and
generate four figures that answer the supervisor's question
*"how can we show that recipes aren't prematurely eliminated?"*:

  1. outputs/figure_top_recipes.png
        Line plot of the top-5 recipes' probability across steps.
  2. outputs/figure_active_recipes.png
        Number of recipes still above probability thresholds (0.01, 0.001,
        0.0001) per step — shows "how many recipes are still in play".
  3. outputs/figure_entropy.png
        Entropy (bits) per step, with observation steps and answer steps
        marked. Shows the information trajectory of the session.
  4. outputs/figure_full_distribution.png
        Stacked area plot of every recipe's probability across steps —
        full distribution visible at a glance.

Each figure also shows the event type (observation vs answer) so the
supervisor can see how questions reshape the distribution.

Run:
    python scripts/plot_belief.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


REPO_ROOT    = Path(__file__).resolve().parent.parent
SESSIONS_DIR = REPO_ROOT / "outputs" / "sessions"

# These are resolved at runtime from the session directory (set in main).
# The module-level constants are kept as backwards-compat defaults.
CSV_PATH       = REPO_ROOT / "outputs" / "belief_history.csv"
REDUNDANCY_CSV = REPO_ROOT / "outputs" / "question_redundancy.csv"
OUT_DIR        = REPO_ROOT / "outputs"


def _pick_latest_session_dir() -> Path | None:
    """Return the most recently modified session folder, or None."""
    if not SESSIONS_DIR.exists():
        return None
    candidates = [p for p in SESSIONS_DIR.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ─────────────────────────────────────────────────────────────────────────
# CSV loading
# ─────────────────────────────────────────────────────────────────────────

def load_csv(path: Path):
    """Return (rows, recipe_names). Each row is a dict with float values
    for probabilities and the step/entropy/etc metadata."""
    if not path.exists():
        print(f"CSV not found: {path}\nRun orchestrator_v2_vlm.py first.",
              file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # cast numeric fields
            r["step"] = int(r["step"])
            r["entropy"] = float(r["entropy"])
            r["ig"] = float(r["ig"])
            r["top_prob"] = float(r["top_prob"])
            for k in list(r.keys()):
                if k.startswith("p::"):
                    r[k] = float(r[k])
            rows.append(r)

    if not rows:
        print("CSV is empty.", file=sys.stderr)
        sys.exit(1)

    recipe_names = [k[3:] for k in rows[0].keys() if k.startswith("p::")]
    return rows, recipe_names


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def step_index(rows: list[dict]) -> list[int]:
    """Return a continuous 0..N-1 index for plotting (CSV step can repeat
    if a window has both an observation and an answer)."""
    return list(range(len(rows)))


def event_labels(rows: list[dict]) -> list[str]:
    """Short label per row used as x-tick text."""
    out: list[str] = []
    for r in rows:
        s = r["step"]
        t = r["type"]
        if t == "initial":
            out.append("init")
        elif t == "observation":
            out.append(f"W{s} obs")
        elif t == "answer_skipped":
            out.append(f"W{s} skip")
        elif t in ("answer_yes", "answer_no"):
            # Polar condition distinguishes yes/no in the event type.
            tag = "yes" if t == "answer_yes" else "no"
            out.append(f"W{s} {tag}")
        else:
            # "answer" (wh) and any future event type
            out.append(f"W{s} ans")
    return out


def mark_event_lines(ax, rows: list[dict]):
    """
    Faint vertical lines at every step where a question was asked, with
    different colours to distinguish:
      - answer / answer_yes / answer_no → crimson (belief updated)
      - answer_skipped                  → goldenrod (question asked but the
                                          cook's reply didn't move belief)
    """
    for i, r in enumerate(rows):
        t = r["type"]
        if t in ("answer", "answer_yes", "answer_no"):
            ax.axvline(i, color="crimson", alpha=0.25,
                       linestyle="--", linewidth=1)
        elif t == "answer_skipped":
            ax.axvline(i, color="goldenrod", alpha=0.35,
                       linestyle=":", linewidth=1.4)


# ─────────────────────────────────────────────────────────────────────────
# Figure 1 — top recipes over time
# ─────────────────────────────────────────────────────────────────────────

def figure_top_recipes(rows, recipe_names, k: int = 5):
    # Pick the top-k recipes by final probability
    last = rows[-1]
    by_final = sorted(
        recipe_names, key=lambda n: -last[f"p::{n}"]
    )[:k]

    xs = step_index(rows)
    labels = event_labels(rows)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name in by_final:
        ys = [r[f"p::{name}"] for r in rows]
        ax.plot(xs, ys, marker="o", label=name, linewidth=2)

    mark_event_lines(ax, rows)
    ax.set_xlabel("Session step")
    ax.set_ylabel("Probability")
    ax.set_title(f"Top {k} recipes — probability across session")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(0, 1.0)

    out = OUT_DIR / "figure_top_recipes.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2 — number of recipes still in play
# ─────────────────────────────────────────────────────────────────────────

def figure_active_recipes(rows, recipe_names,
                          thresholds=(0.01, 0.001, 0.0001)):
    xs = step_index(rows)
    labels = event_labels(rows)

    fig, ax = plt.subplots(figsize=(11, 6))
    for thr in thresholds:
        counts = [
            sum(1 for n in recipe_names if r[f"p::{n}"] > thr)
            for r in rows
        ]
        ax.plot(xs, counts, marker="o", linewidth=2,
                label=f"p > {thr}")

    mark_event_lines(ax, rows)
    ax.set_xlabel("Session step")
    ax.set_ylabel("Recipes above probability threshold")
    ax.set_title(
        f"How many recipes remain plausible? (of {len(recipe_names)} total)"
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_ylim(0, len(recipe_names) + 1)

    out = OUT_DIR / "figure_active_recipes.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3 — entropy curve
# ─────────────────────────────────────────────────────────────────────────

def figure_entropy(rows):
    xs = step_index(rows)
    labels = event_labels(rows)
    ys = [r["entropy"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(xs, ys, marker="o", linewidth=2, color="steelblue")

    # Annotate by event type. The hollow gold X marker for "answer_skipped"
    # is what the supervisor needs to see — those are windows where we DID
    # ask the cook a question but the reply was OFF_TOPIC / DOESNT_APPLY /
    # DONT_KNOW / NEGATIVE, so the belief didn't move. Without this marker
    # those questions are invisible on the plot.
    seen_labels: set[str] = set()
    for i, r in enumerate(rows):
        t = r["type"]
        if t == "observation":
            ax.scatter(i, r["entropy"], s=70, color="steelblue", zorder=3)
        elif t in ("answer", "answer_yes"):
            lbl = "answer (belief updated)"
            ax.scatter(i, r["entropy"], s=90, color="crimson", zorder=3,
                       marker="D",
                       label=lbl if lbl not in seen_labels else None)
            seen_labels.add(lbl)
        elif t == "answer_no":
            lbl = "answer NO (negation)"
            ax.scatter(i, r["entropy"], s=90, color="darkorange", zorder=3,
                       marker="v",
                       label=lbl if lbl not in seen_labels else None)
            seen_labels.add(lbl)
        elif t == "answer_skipped":
            lbl = "asked but skipped (no belief update)"
            ax.scatter(i, r["entropy"], s=110, color="goldenrod", zorder=3,
                       marker="X", edgecolors="black", linewidths=1.2,
                       label=lbl if lbl not in seen_labels else None)
            seen_labels.add(lbl)

    mark_event_lines(ax, rows)

    # Add a horizontal line at max entropy for reference
    if rows:
        # max entropy ≈ log2(N) — read from initial entropy of uniform prior
        max_h = max(r["entropy"] for r in rows[:1]) if rows else 0
        ax.axhline(max_h, color="grey", linestyle=":", linewidth=1,
                   label=f"initial entropy ({max_h:.2f} bits)")

    ax.set_xlabel("Session step")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title(
        "Belief entropy across session\n"
        "blue dot = observation, red diamond = answer, "
        "orange ▽ = answer NO, gold X = asked but skipped"
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = OUT_DIR / "figure_entropy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ─────────────────────────────────────────────────────────────────────────
# Figure 4 — full stacked distribution
# ─────────────────────────────────────────────────────────────────────────

def figure_full_distribution(rows, recipe_names):
    # Sort recipes by final probability so the biggest stack on top
    last = rows[-1]
    sorted_names = sorted(recipe_names, key=lambda n: -last[f"p::{n}"])

    xs = step_index(rows)
    labels = event_labels(rows)
    series = [
        [r[f"p::{n}"] for r in rows]
        for n in sorted_names
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.stackplot(xs, *series, labels=sorted_names, alpha=0.85)

    mark_event_lines(ax, rows)
    ax.set_xlabel("Session step")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(
        "Full belief distribution across session "
        "(stacked — every recipe visible)"
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    # Legend outside so it doesn't cover the stack
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=7, ncol=1)

    out = OUT_DIR / "figure_full_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def figure_question_redundancy():
    """
    Per-question bar chart: IG_Q vs measured redundancy with the next
    observation, plus unique contribution. Only rendered if the
    redundancy CSV exists.
    """
    if not REDUNDANCY_CSV.exists():
        print(f"  (no {REDUNDANCY_CSV.name}; skipping redundancy plot)")
        return

    rows = []
    with open(REDUNDANCY_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # Cast numerics; blank cells stay as None
            for k in ("ig_q", "real_ig_next_obs", "shadow_ig_next_obs",
                      "redundancy", "unique_value", "predicted_eig"):
                v = r.get(k, "").strip()
                r[k] = float(v) if v not in ("", "None", None) else None
            rows.append(r)

    if not rows:
        print("  (redundancy CSV empty; skipping)")
        return

    xs = list(range(len(rows)))

    # Tag rows so we can render skipped questions differently. A row is
    # "skipped" when the answer type isn't an actual belief-changing one
    # (off_topic, doesnt_apply, dont_know, negative, "skipped") and the
    # ig_q value is zero. The redundancy is None for those rows because
    # the question never moved the belief, so no overlap can be measured.
    SKIPPED_ANSWER_TYPES = {
        "off_topic", "doesnt_apply", "dont_know", "negative", "skipped",
    }
    is_skipped = [
        (r.get("answer_type") or "") in SKIPPED_ANSWER_TYPES
        for r in rows
    ]

    labels = []
    for r, sk in zip(rows, is_skipped):
        suffix = ""
        if sk:
            t = (r.get("answer_type") or "skipped").replace("_", " ")
            suffix = f"\n[{t}]"
        labels.append(f"W{r['window']}{suffix}")

    ig_q       = [r["ig_q"] or 0.0 for r in rows]
    redundancy = [r["redundancy"] or 0.0 for r in rows]
    unique     = [r["unique_value"] or 0.0 for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.27

    # IG_Q bar: colour depends on whether the question actually delivered
    # information or was skipped. Skipped bars are drawn as hatched gold
    # outlines so the supervisor can immediately see "we asked but learned
    # nothing" at those windows.
    answered_xs = [x for x, sk in zip(xs, is_skipped) if not sk]
    answered_ig = [v for v, sk in zip(ig_q, is_skipped) if not sk]
    skipped_xs  = [x for x, sk in zip(xs, is_skipped) if sk]
    skipped_eig = [
        (r.get("predicted_eig") or 0.0)
        for r, sk in zip(rows, is_skipped) if sk
    ]

    if answered_xs:
        ax.bar([x - width for x in answered_xs], answered_ig, width,
               label="IG_Q (answer alone)", color="steelblue")
    if skipped_xs:
        # Show predicted EIG as a hollow gold bar so the reader sees
        # what the planner expected from the question that ended up
        # being skipped. The actual realised IG_Q is zero (no bar).
        ax.bar([x - width for x in skipped_xs], skipped_eig, width,
               facecolor="none", edgecolor="goldenrod", linewidth=1.8,
               hatch="//",
               label="asked but skipped (predicted EIG shown)")

    # Redundancy and unique value are only meaningful for non-skipped
    # questions — those have measurable next-observation comparisons.
    answered_red = [v for v, sk in zip(redundancy, is_skipped) if not sk]
    answered_uni = [v for v, sk in zip(unique, is_skipped) if not sk]
    if answered_xs:
        ax.bar(answered_xs, answered_red, width,
               label="Redundancy with next obs", color="darkorange")
        ax.bar([x + width for x in answered_xs], answered_uni, width,
               label="Unique value", color="seagreen")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Question event")
    ax.set_ylabel("Bits")
    ax.set_title(
        "Per-question information value\n"
        "Redundancy = how much of IG_Q the next observation would have "
        "delivered anyway. Hollow gold bars = questions asked but skipped "
        "(predicted EIG shown; realised IG_Q was zero)."
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    out = OUT_DIR / "figure_question_redundancy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot belief-trajectory figures for one orchestrator session. "
            "Reads belief_history.csv (and optionally question_redundancy.csv) "
            "from the given session directory and writes figures alongside."
        ),
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help=(
            "Path to one outputs/sessions/<session_id>/ folder. "
            "If omitted, the most recent session folder is used. "
            "Backwards-compat: pass outputs/ to use the top-level CSVs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    global CSV_PATH, REDUNDANCY_CSV, OUT_DIR

    args = parse_args()

    if args.session_dir is not None:
        session_dir = args.session_dir.resolve()
    else:
        latest = _pick_latest_session_dir()
        if latest is None:
            # Fall back to the legacy top-level outputs/ layout.
            session_dir = REPO_ROOT / "outputs"
            print("[plot_belief] No session dirs found — using top-level outputs/")
        else:
            session_dir = latest
            print(f"[plot_belief] Auto-selected latest session: {session_dir.name}")

    if not session_dir.exists():
        print(f"Session directory not found: {session_dir}", file=sys.stderr)
        sys.exit(1)

    CSV_PATH       = session_dir / "belief_history.csv"
    REDUNDANCY_CSV = session_dir / "question_redundancy.csv"
    OUT_DIR        = session_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {CSV_PATH}")
    rows, recipe_names = load_csv(CSV_PATH)
    print(f"  {len(rows)} events, {len(recipe_names)} recipes")
    print()

    print("Generating figures...")
    figure_top_recipes(rows, recipe_names, k=5)
    figure_active_recipes(rows, recipe_names)
    figure_entropy(rows)
    figure_full_distribution(rows, recipe_names)
    figure_question_redundancy()
    print()
    print(f"Done. Figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
