"""
test_scene_graph.py
-------------------
Mock dynamic scene graph simulating a cooking session clip by clip.
Simulates someone making a creamy cucumber salad — but early windows
are ambiguous (shared ingredients) and only later windows reveal
the discriminating ingredient (sour cream).

Each window represents one video clip fed through the VLM.
"""

from src.dynamic_graph import ObservationHistory, WindowObservation

def build_mock_session() -> list[ObservationHistory]:
    """
    Returns a list of ObservationHistory objects, one per window,
    each representing the cumulative state after that clip.

    Simulates: creamy cucumber salad session
      Window 1: slice cucumber       → ambiguous (everyone slices cucumber)
      Window 2: chop onion           → still ambiguous (everyone uses onion)
      Window 3: whisk sour cream     → discriminating! only creamy recipe uses this
      Window 4: toss and cover       → finishing steps, all recipes do this
    """

    windows = [
        WindowObservation(
            window=1,
            clip_path="data/clip_01_slice_cucumber.mp4",
            summary="person slices cucumber on a cutting board",
            verb="slice",
            object_target="cucumber",
            ingredients=["cucumber"],
            tools=["knife", "cutting board"],
            object_states={"cucumber": "sliced"}
        ),
        WindowObservation(
            window=2,
            clip_path="data/clip_02_chop_onion.mp4",
            summary="person chops onion on a cutting board",
            verb="chop",
            object_target="onion",
            ingredients=["onion"],
            tools=["knife", "cutting board"],
            object_states={"onion": "chopped"}
        ),
        WindowObservation(
            window=3,
            clip_path="data/clip_03_whisk_sour_cream.mp4",
            summary="person whisks sour cream with sugar and vinegar in a bowl",
            verb="whisk",
            object_target="sour cream mixture",
            ingredients=["sour cream", "white sugar", "white vinegar"],
            tools=["bowl", "whisk"],
            object_states={"sour cream": "whisked", "dressing": "creamy and mixed"}
        ),
        WindowObservation(
            window=4,
            clip_path="data/clip_04_toss_cover.mp4",
            summary="person tosses salad and covers the bowl",
            verb="combine",
            object_target="salad",
            ingredients=[],
            tools=["bowl"],
            object_states={"salad": "chilling"}
        ),
    ]

    # Build cumulative histories — one per window
    histories = []
    history = ObservationHistory()
    for obs in windows:
        history.add(obs)
        # Deep copy via dict round-trip so each snapshot is independent
        snapshot = ObservationHistory()
        for o in history.observations:
            snapshot.add(o)
        histories.append(snapshot)

    return histories


if __name__ == "__main__":
    import json
    histories = build_mock_session()
    for i, h in enumerate(histories, start=1):
        d = h.to_dict()
        print(f"\n=== After window {i} ===")
        print(f"  ingredients_seen : {d['derived']['ingredients_seen']}")
        print(f"  actions_so_far   : {d['derived']['actions_so_far']}")
        print(f"  latest_states    : {d['derived']['latest_states']}")
