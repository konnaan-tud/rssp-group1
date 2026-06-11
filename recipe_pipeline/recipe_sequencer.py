from __future__ import annotations

import json
from pathlib import Path

from observation_pipeline.embedder import SentenceEmbedder


def _load_graphs(graph_dir: Path) -> list[dict]:
    graphs: list[dict] = []
    for path in sorted(graph_dir.glob("*.json")):
        if path.name == "recipe_sequences.json":
            continue
        with path.open("r", encoding="utf-8") as handle:
            graphs.append(json.load(handle))
    return graphs


def build_recipe_sequences(
    graph_dir: str | Path,
    output_path: str | Path,
    embedder: SentenceEmbedder | None = None,
) -> dict[str, dict]:
    graph_dir = Path(graph_dir)
    output_path = Path(output_path)
    embedder = embedder or SentenceEmbedder()

    sequences: dict[str, dict] = {}
    for graph in _load_graphs(graph_dir):
        sentences = graph.get("scene_sentences", [])
        sequences[graph["recipe_id"]] = {
            "recipe_name": graph.get("recipe_name", graph["recipe_id"]),
            "sentences": sentences,
            "embeddings": [embedder.embed(sentence) for sentence in sentences],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(sequences, handle, indent=2)
    return sequences
