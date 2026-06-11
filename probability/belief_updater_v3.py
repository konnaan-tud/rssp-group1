from __future__ import annotations

import json
import math
from pathlib import Path

from observation_pipeline.embedder import SentenceEmbedder
from probability.dtw import dtw_distance


class BeliefUpdaterV3:
    def __init__(self, recipe_sequences: dict[str, dict], embedder: SentenceEmbedder | None = None):
        self.recipe_sequences = recipe_sequences
        self.embedder = embedder or SentenceEmbedder()
        self.observation_sentences: list[str] = []
        self.belief = self._uniform_belief()

    @classmethod
    def from_recipe_directory(
        cls,
        graph_dir: str | Path,
        sequence_path: str | Path,
        embedder: SentenceEmbedder | None = None,
    ) -> "BeliefUpdaterV3":
        sequence_path = Path(sequence_path)
        if not sequence_path.exists():
            raise FileNotFoundError(f"Missing recipe sequence file: {sequence_path}")
        with sequence_path.open("r", encoding="utf-8") as handle:
            recipe_sequences = json.load(handle)
        return cls(recipe_sequences=recipe_sequences, embedder=embedder)

    def _uniform_belief(self) -> dict[str, float]:
        count = max(len(self.recipe_sequences), 1)
        weight = 1.0 / count
        return {recipe_id: weight for recipe_id in self.recipe_sequences}

    def update(self, observation_sentences: list[str]) -> dict[str, float]:
        self.observation_sentences.extend(observation_sentences)
        observation_embeddings = [self.embedder.embed(sentence) for sentence in self.observation_sentences]

        scores: dict[str, float] = {}
        for recipe_id, recipe in self.recipe_sequences.items():
            distance = dtw_distance(observation_embeddings, recipe.get("embeddings", []))
            scores[recipe_id] = math.exp(-distance)

        total = sum(scores.values()) or 1.0
        self.belief = {recipe_id: score / total for recipe_id, score in scores.items()}
        return self.belief

    def top_recipe(self) -> tuple[str, float]:
        return max(self.belief.items(), key=lambda item: item[1])
