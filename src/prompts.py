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

    "describe_inna": (
            "You are watching a short cooking video clip. "
            "Describe only what is clearly visible in the clip. "
            "Do not infer hidden ingredients, recipe names, or actions that are not directly shown. "
            "If an object or ingredient is unclear, say 'unclear' instead of guessing. "
            "Use at most 5 bullet points. "
            "Do not repeat the same action."
    ),

    "evidence_vs_guess": (
        "Analyze the cooking video clip. Separate what is clearly visible from what is uncertain. "
        "Do not leave all fields empty unless the video is blank or unreadable. "
        "Use simple generic descriptions when exact objects are unclear, for example "
        "'person handling an object', 'person standing near counter', or 'person moving hand'.\n\n"
        "Use exactly this format:\n"
        "- direct_visual_evidence: [visible actions or objects, using generic labels if needed]\n"
        "- uncertain_observations: [things that may be visible but are not clear]\n"
        "- possible_inferences: [reasonable guesses, marked as guesses]\n"
        "- do_not_treat_as_fact: [claims that would require guessing]"
    ),

    "handled_objects": (
        "You are watching a short cooking video clip. "
        "Focus only on objects that the person touches, picks up, carries, removes, places down, or uses. "
        "List every distinct handled object, even if it is visible only briefly. "
        "Do not summarize the scene as a story. "
        "Do not ignore secondary objects. "
        "If you are unsure what an object is, describe its appearance and write 'unclear'.\n\n"
        "Use exactly this format:\n"
        "- handled_objects:\n"
        "  1. object: <name or unclear>\n"
        "     action: <picked up / removed / placed / carried / used / unclear>\n"
        "     visual_description: <short description of shape/color/location>\n"
        "     confidence: <low/medium/high>\n"
        "  2. object: <name or unclear>\n"
        "     action: <picked up / removed / placed / carried / used / unclear>\n"
        "     visual_description: <short description of shape/color/location>\n"
        "     confidence: <low/medium/high>\n"
        "- missed_or_unclear_objects: [objects that may have been handled but are hard to identify]"
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
