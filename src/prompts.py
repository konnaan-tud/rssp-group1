"""Prompt templates for the VLM-based cooking observer baseline.

Add more prompts here as we iterate. Each prompt is a plain string that gets
sent as the text part of a Qwen2.5-VL message.
"""

PROMPTS = {
    "describe": (
        "You are watching a video of someone cooking. Describe in detail what "
        "the person is doing in this video clip — the actions, objects, "
        "ingredients, and tools that are visible. Do not guess what recipe is "
        "being prepared yet, only describe what you actually observe."
    ),

    "recipe_guess": (
        "You are watching a video of someone cooking. Based on the actions, "
        "ingredients, and tools you observe:\n"
        "1. Briefly describe what the person is doing.\n"
        "2. List your top three guesses for what recipe is being prepared, "
        "in order of likelihood. For each, give a one-line reason.\n"
        "3. State which observed details would help confirm or rule out each "
        "guess."
    ),

    "structured_state": (
        "You are observing a person cooking. Report what you see in the "
        "following structured format and nothing else:\n"
        "- objects_in_scene: [list]\n"
        "- tools_in_use: [list]\n"
        "- ingredients_visible: [list]\n"
        "- current_action_verb: <verb>\n"
        "- current_action_object: <object>\n"
        "- recipe_step_inference: <short phrase>"
    ),
}


def get_prompt(name: str) -> str:
    if name not in PROMPTS:
        raise KeyError(
            f"Prompt {name!r} not found. Available: {list(PROMPTS)}"
        )
    return PROMPTS[name]
