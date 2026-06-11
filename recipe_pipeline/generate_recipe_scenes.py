from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _split_field(value: str) -> list[str]:
    if not value:
        return []
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(part).strip() for part in parsed if str(part).strip()]
    parts = re.split(r"[|,;/]", value)
    return [part.strip() for part in parts if part.strip()]


def _recipe_title(row: dict[str, str]) -> str:
    return (
        row.get("recipe_title")
        or row.get("title")
        or row.get("recipe")
        or row.get("name")
        or row.get("normalized_recipe_name")
        or "unknown recipe"
    )


def build_scene_sentences(row: dict[str, str]) -> list[str]:
    title = _recipe_title(row)
    ingredients = _split_field(
        row.get("clean_ingredients")
        or row.get("core_ingredients_for_similarity")
        or row.get("ingredients")
        or row.get("ingredient_list")
        or row.get("NER")
        or ""
    )
    directions = _split_field(
        row.get("directions")
        or row.get("direction_text")
        or row.get("steps")
        or row.get("instructions")
        or row.get("action_sequence")
        or ""
    )

    sentences: list[str] = []
    if ingredients:
        sentences.append(f"Prepare ingredients for {title}: {', '.join(ingredients)}.")
    for index, step in enumerate(directions, start=1):
        sentences.append(f"Step {index} of {title}: {step}.")
    if not sentences:
        sentences.append(f"Recipe summary for {title}.")
    return sentences


def row_to_graph(row: dict[str, str]) -> dict:
    title = _recipe_title(row)
    graph_id = _slugify(title)
    scene_sentences = build_scene_sentences(row)
    ingredients = _split_field(
        row.get("clean_ingredients")
        or row.get("core_ingredients_for_similarity")
        or row.get("ingredients")
        or row.get("ingredient_list")
        or row.get("NER")
        or ""
    )

    return {
        "recipe_id": graph_id,
        "recipe_name": title,
        "scene_sentences": scene_sentences,
        "all_ingredients": ingredients,
        "metadata": {
            "source_columns": sorted(row.keys()),
        },
    }


def generate_recipe_scene_graphs(dataset_path: str | Path, output_dir: str | Path) -> list[Path]:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for existing_file in output_dir.glob("*.json"):
        existing_file.unlink()

    written_files: list[Path] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            graph = row_to_graph(row)
            output_path = output_dir / f"{graph['recipe_id']}.json"
            with output_path.open("w", encoding="utf-8") as out:
                json.dump(graph, out, indent=2, ensure_ascii=False)
            written_files.append(output_path)

    return written_files
