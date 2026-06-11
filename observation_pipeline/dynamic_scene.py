from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClipObservation:
    clip_id: str
    ingredients: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class DynamicScene:
    clip_id: str
    sentences: list[str]


class DynamicSceneBuilder:
    """Convert clip-level observations into sentence-based scene descriptions."""

    def build(self, observation: ClipObservation) -> DynamicScene:
        sentences: list[str] = []
        if observation.ingredients:
            sentences.append(
                f"Clip {observation.clip_id} shows ingredients: "
                + ", ".join(observation.ingredients)
                + "."
            )
        if observation.actions:
            sentences.append(
                f"Clip {observation.clip_id} shows actions: "
                + ", ".join(observation.actions)
                + "."
            )
        if observation.notes:
            sentences.append(observation.notes.strip())
        if not sentences:
            sentences.append(f"Clip {observation.clip_id} contains no extracted observations.")
        return DynamicScene(clip_id=observation.clip_id, sentences=sentences)
