"""Per-window observation records of a cooking session.

For each video clip, the VLM emits a single structured observation describing
what it saw: action, target, ingredients, tools, state changes. We append
those observations to a list — no mutable graph, no deltas, no add/modify/remove.

The "current state" of the kitchen is derived from this list when needed,
e.g. by taking the latest observation's action, or unioning ingredients seen
so far.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional


# Constrained vocabularies for the VLM (also useful for downstream matching).
HUMAN_VERBS = {
    "chop", "dice", "slice", "saute", "transfer", "stir", "mix", "pour",
    "add", "hold", "season", "boil", "whisk", "drain", "fry", "simmer",
    "combine", "rinse", "peel", "grate", "idle",
}
OBJECT_STATES = {
    "raw", "chopped", "diced", "sliced", "cooked", "browning", "boiling",
    "simmering", "drained", "rinsed", "peeled", "mixed",
}


# ---------- Observation record ----------

@dataclass
class WindowObservation:
    """What the VLM saw in one video clip."""
    window: int
    clip_path: str
    summary: str = ""                     # one-sentence description
    verb: str = ""                        # primary action: chop, saute, ...
    object_target: str = ""               # what is being acted on
    ingredients: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    object_states: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ObservationHistory:
    """Append-only list of per-window observations. This is our 'dynamic state'."""

    def __init__(self) -> None:
        self.observations: list[WindowObservation] = []

    def add(self, obs: WindowObservation) -> None:
        self.observations.append(obs)

    # Convenience accessors built on top of the list ---------------------

    def latest(self) -> Optional[WindowObservation]:
        return self.observations[-1] if self.observations else None

    def ingredients_seen(self) -> list[str]:
        seen: list[str] = []
        for o in self.observations:
            for ing in o.ingredients:
                if ing not in seen:
                    seen.append(ing)
        return seen

    def tools_seen(self) -> list[str]:
        seen: list[str] = []
        for o in self.observations:
            for t in o.tools:
                if t not in seen:
                    seen.append(t)
        return seen

    def actions_so_far(self) -> list[str]:
        return [o.verb for o in self.observations if o.verb]

    def latest_states(self) -> dict[str, str]:
        """For each object, return the most recent state observed across windows."""
        out: dict[str, str] = {}
        for o in self.observations:
            out.update(o.object_states)
        return out

    def compact_context(self, recent_windows: int = 5) -> str:
        """Render a compact prompt context while preserving full history in storage."""
        if not self.observations:
            return "(nothing observed yet)"

        recent = self.observations[-recent_windows:]

        ingredients = ", ".join(self.ingredients_seen()) or "—"
        tools = ", ".join(self.tools_seen()) or "—"

        latest_states = self.latest_states()
        if latest_states:
            state_text = ", ".join(
                f"{obj}={state}" for obj, state in latest_states.items()
            )
        else:
            state_text = "—"

        action_lines = []
        for o in recent:
            if not o.verb:
                continue
            action = o.verb
            if o.object_target:
                action = f"{action} {o.object_target}"
            action_lines.append(f"* {action}")
        if not action_lines:
            action_lines.append("* —")

        observation_lines = []
        for o in recent:
            summary = o.summary
            if not summary:
                summary = f"verb={o.verb or '—'}; target={o.object_target or '—'}"
            observation_lines.append(
                f"* window {o.window}: {summary}"
            )

        return "\n".join([
            "Kitchen state:",
            f"* ingredients_seen: {ingredients}",
            f"* tools_seen: {tools}",
            f"* latest_states: {state_text}",
            "",
            "Recent actions:",
            *action_lines,
            "",
            f"Recent observations (last {len(recent)} windows):",
            *observation_lines,
        ])

    def to_dict(self) -> dict:
        return {
            "observations": [o.to_dict() for o in self.observations],
            "derived": {
                "ingredients_seen": self.ingredients_seen(),
                "tools_seen": self.tools_seen(),
                "actions_so_far": self.actions_so_far(),
                "latest_states": self.latest_states(),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------- VLM prompting ----------

PROMPT_TEMPLATE = """You are observing a cooking session through a fixed camera.
You will see one short video clip per call. Describe ONLY what you can directly see
in the frames — do not invent objects, ingredients, or actions.

For context, here is what has been seen in earlier clips of this session:
{history_summary}

Recipe context: {recipe_summary}

Output ONLY a single JSON object, no commentary, in this exact shape:

{{
  "summary":       "<one short sentence describing what is happening>",
  "verb":          "<single primary action the person performs>",
  "object_target": "<what the person is acting on>",
  "ingredients":   ["<ingredient>", ...],
  "tools":         ["<tool>", ...],
  "object_states": {{"<object>": "<state>", ...}}
}}

Constraints:
- Use a verb from this set when possible: {verbs}
- Use states from this set when possible: {states}
- Do NOT include colors, sizes, shapes, lighting, camera angles, or backgrounds.
- Use lowercase singular nouns for ingredients and tools (e.g. "onion", not "Onions").
- If something is not visible, leave its list empty or its field as an empty string.

Output the JSON only.
"""


def build_prompt(history: ObservationHistory, recipe_summary: str = "Unknown.") -> str:
    """Compose the VLM prompt for the next observation window."""
    history_summary = history.compact_context()

    return PROMPT_TEMPLATE.format(
        history_summary=history_summary,
        recipe_summary=recipe_summary,
        verbs=", ".join(sorted(HUMAN_VERBS)),
        states=", ".join(sorted(OBJECT_STATES)),
    )


# ---------- VLM response parsing ----------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_observation(raw_response: str, window: int, clip_path: str) -> WindowObservation:
    """Parse the VLM's JSON response into a WindowObservation. Forgiving:
    missing or malformed fields fall back to empty values."""
    text = _FENCE_RE.sub("", raw_response).strip()
    start = text.find("{")
    end = text.rfind("}")
    parsed: dict = {}
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if not isinstance(parsed, dict):
                parsed = {}
        except json.JSONDecodeError:
            parsed = {}

    return WindowObservation(
        window=window,
        clip_path=clip_path,
        summary=_str(parsed.get("summary")),
        verb=_str(parsed.get("verb")).lower(),
        object_target=_str(parsed.get("object_target")).lower(),
        ingredients=_strlist(parsed.get("ingredients")),
        tools=_strlist(parsed.get("tools")),
        object_states=_strdict(parsed.get("object_states")),
    )


def _str(v) -> str:
    return v.strip() if isinstance(v, str) else ""


def _strlist(v) -> list[str]:
    if not isinstance(v, list):
        return []
    return [s.strip().lower() for s in v if isinstance(s, str) and s.strip()]


def _strdict(v) -> dict[str, str]:
    if not isinstance(v, dict):
        return {}
    return {
        str(k).strip().lower(): str(val).strip().lower()
        for k, val in v.items()
        if isinstance(val, (str, int, float))
    }


# ---------- smoke test ----------

if __name__ == "__main__":
    history = ObservationHistory()

    # Simulate three VLM responses, each one structured per clip.
    fake_responses = [
        # window 1: chopping onions
        '{"summary": "Person chops an onion on a cutting board.", '
        '"verb": "chop", "object_target": "onion", '
        '"ingredients": ["onion"], "tools": ["knife", "cutting board"], '
        '"object_states": {"onion": "diced"}}',
        # window 2: sauteing those onions
        '```json\n{"summary": "Person stirs onions in a hot pan.", '
        '"verb": "saute", "object_target": "onion", '
        '"ingredients": ["onion", "oil"], "tools": ["pan", "spatula"], '
        '"object_states": {"onion": "browning"}}\n```',
        # window 3: minced meat
        '{"summary": "Person stirs minced meat in a pan.", '
        '"verb": "saute", "object_target": "minced meat", '
        '"ingredients": ["minced meat", "oil"], "tools": ["pan", "spatula"], '
        '"object_states": {"minced meat": "browning"}}',
    ]

    for i, raw in enumerate(fake_responses, start=1):
        obs = parse_observation(raw, window=i, clip_path=f"data/clip_{i}.mp4")
        history.add(obs)

    print(history.to_json())

    # Show what the next prompt would look like:
    print("\n----- Next prompt (history rendered for VLM) -----")
    print(build_prompt(history, recipe_summary="Spaghetti bolognese early steps"))
