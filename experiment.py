"""Run sequential recipe-clip experiments.

Expected data layout
--------------------
Each recipe is one folder under ``data/`` and contains short video clips in
sequential order, for example::

    data/recipe_1/
        01.mp4
        02.mp4
        03.mp4

For each clip, this script:
1. calls the existing dynamic-loop VLM function to parse one observation;
2. appends that observation to the existing ObservationHistory;
3. runs the existing BeliefUpdater over the observation history;
4. calls the existing questioning pipeline to generate clarification questions;
5. records entropy / information-gain bookkeeping.

Important limitation
--------------------
The current repository does not contain a function that obtains clarification
answers from a user/oracle, parses those answers, and applies them back into the
ObservationHistory or BeliefUpdater. That part is therefore left commented as a
TODO below instead of inventing new functionality here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from belief_updater import BeliefUpdater
from src.dynamic_graph import ObservationHistory


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def natural_sort_key(path: Path) -> list[Any]:
    """Sort paths as humans expect: 2.mp4 before 10.mp4."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_recipe_clips(recipe_folder: Path) -> list[Path]:
    """Return clips in sequential filename order from one recipe folder."""
    if not recipe_folder.exists():
        raise FileNotFoundError(f"Recipe folder not found: {recipe_folder}")
    if not recipe_folder.is_dir():
        raise NotADirectoryError(f"Recipe path is not a folder: {recipe_folder}")

    clips = [p for p in recipe_folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    clips = sorted(clips, key=natural_sort_key)
    if not clips:
        raise FileNotFoundError(f"No video clips found in: {recipe_folder}")
    return clips


def entropy_bits(belief: dict[str, float]) -> float:
    """Shannon entropy in bits for an already-normalised belief distribution."""
    return -sum(p * math.log2(p) for p in belief.values() if p > 0)


def information_gain(before: dict[str, float], after: dict[str, float]) -> float:
    """Observed entropy reduction H(before) - H(after)."""
    return entropy_bits(before) - entropy_bits(after)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from an LLM response if possible."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_experiment(
    recipe_folder: Path,
    recipe_summary: str,
    graph_folder: Path,
    output_dir: Path,
    fps: float = 0.5,
    model_name: str = DEFAULT_MODEL,
    question_model: str | None = None,
    generate_questions: bool = True,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """Run one sequential experiment over a single recipe folder."""
    # Lazy imports keep `python experiment.py --help` usable even on machines
    # where the heavy VLM dependencies have not been installed yet. Running the
    # experiment itself still requires `pip install -r requirements.txt`.
    from questioning_pipeline import (
        build_question_prompt,
        load_static_graphs,
        run_qwen_prompt,
    )
    from src.run_dynamic_loop import load_model, run_single_window

    clips = list_recipe_clips(recipe_folder)
    static_graphs = load_static_graphs(graph_folder)

    history = ObservationHistory()
    belief_updater = BeliefUpdater(static_graphs)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {
        "recipe_folder": str(recipe_folder),
        "recipe_summary": recipe_summary,
        "graph_folder": str(graph_folder),
        "clips": [str(c) for c in clips],
        "fps": fps,
        "model": model_name,
        "question_model": question_model,
        "generate_questions": generate_questions,
        "max_new_tokens": max_new_tokens,
    })

    print(f"[experiment] loading VLM once for {len(clips)} clips", flush=True)
    model, processor, device = load_model(model_name)

    results: list[dict[str, Any]] = []

    for window_idx, clip_path in enumerate(clips, start=1):
        print(f"\n========== Clip {window_idx}/{len(clips)}: {clip_path.name} ==========", flush=True)
        t0 = time.time()

        belief_before_video = dict(belief_updater.belief)

        # Existing functionality from src/run_dynamic_loop.py.
        observation, raw_vlm_response = run_single_window(
            model=model,
            processor=processor,
            device=device,
            history=history,
            window_idx=window_idx,
            video_path=str(clip_path),
            recipe_summary=recipe_summary,
            fps=fps,
            max_new_tokens=max_new_tokens,
        )
        history.add(observation)

        belief_after_video = belief_updater.update(history)
        video_information_gain = information_gain(belief_before_video, belief_after_video)

        question_prompt = None
        raw_question_response = None
        parsed_questions = None
        belief_after_question = None
        question_information_gain = None

        if generate_questions:
            # Existing functionality from questioning_pipeline.py.
            question_prompt = build_question_prompt(
                static_graphs=static_graphs,
                belief_state=belief_after_video,
            )
            raw_question_response = run_qwen_prompt(
                question_prompt,
                model_name=question_model or model_name,
            )
            parsed_questions = parse_json_object(raw_question_response)

            # TODO: Missing existing functionality.
            # The repository currently has question generation, but not a callable
            # component that obtains a user/oracle answer, parses that answer into
            # observations/evidence, and re-runs BeliefUpdater on that evidence.
            # When that exists, the intended flow is:
            #
            #   answer = ask_oracle(parsed_questions["questions"][0], clip_path, history)
            #   updated_history = apply_question_answer(history, answer)
            #   belief_after_question = belief_updater.update(updated_history)
            #   question_information_gain = information_gain(
            #       belief_after_video,
            #       belief_after_question,
            #   )
            #
            # Until then, belief_after_question and question_information_gain remain None.

        elapsed = time.time() - t0
        summary = belief_updater.summary()

        record = {
            "window": window_idx,
            "clip": str(clip_path),
            "elapsed_seconds": elapsed,
            "raw_vlm_response": raw_vlm_response,
            "parsed_observation": observation.to_dict(),
            "history_after": history.to_dict(),
            "belief_before_video": belief_before_video,
            "belief_after_video": belief_after_video,
            "video_information_gain_bits": video_information_gain,
            "question_prompt": question_prompt,
            "raw_question_response": raw_question_response,
            "parsed_questions": parsed_questions,
            "belief_after_question": belief_after_question,
            "question_information_gain_bits": question_information_gain,
            "belief_summary": summary,
        }
        results.append(record)

        write_json(output_dir / f"window_{window_idx:02d}.json", record)

        print(f"[experiment] top recipe: {summary['top_recipe']} ({summary['top_prob']:.4f})", flush=True)
        print(f"[experiment] entropy: {summary['entropy']:.4f} bits", flush=True)
        print(f"[experiment] video IG: {video_information_gain:.4f} bits", flush=True)
        if parsed_questions:
            first_question = parsed_questions.get("questions", [{}])[0].get("question")
            if first_question:
                print(f"[experiment] best generated question: {first_question}", flush=True)

    final_payload = {
        "recipe_folder": str(recipe_folder),
        "recipe_summary": recipe_summary,
        "final_history": history.to_dict(),
        "final_belief": dict(belief_updater.belief),
        "final_summary": belief_updater.summary(),
        "windows": results,
    }
    write_json(output_dir / "experiment_results.json", final_payload)
    print(f"\n[experiment] wrote {output_dir / 'experiment_results.json'}", flush=True)
    return final_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sequential recipe-clip experiments.")
    parser.add_argument(
        "--recipe-folder",
        required=True,
        type=Path,
        help="Folder containing the sequential clips for one recipe, e.g. data/recipe_1",
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help="Short recipe context sent to the VLM. Defaults to the recipe folder name.",
    )
    parser.add_argument(
        "--graph-folder",
        type=Path,
        default=Path("recipe_graphs/instruct"),
        help="Folder containing candidate static recipe graph JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments"),
        help="Folder where experiment logs/results are written.",
    )
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--question-model",
        default=None,
        help="Optional separate model for question generation. Defaults to --model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--skip-questions",
        action="store_true",
        help="Only run VLM observation + belief update; skip question generation.",
    )
    args = parser.parse_args()

    recipe_summary = args.recipe or args.recipe_folder.name.replace("_", " ")
    output_dir = args.output_dir / args.recipe_folder.name

    try:
        run_experiment(
            recipe_folder=args.recipe_folder,
            recipe_summary=recipe_summary,
            graph_folder=args.graph_folder,
            output_dir=output_dir,
            fps=args.fps,
            model_name=args.model,
            question_model=args.question_model,
            generate_questions=not args.skip_questions,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as exc:
        print(f"[experiment] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
