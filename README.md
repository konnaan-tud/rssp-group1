# Project 1 — Cooking Intent Recognition with VLM Clarification

A VLM-based cooking observer that watches a human prepare a meal, infers the
recipe being prepared from a finite set of candidates, and (eventually) asks
clarification questions when its belief is uncertain. This repository hosts the
working pipeline; the methodology and evaluation are described in the proposal.

## What the pipeline does

1. A short video clip is fed to **Qwen2.5-VL** (7B for experiments, 3B for
   laptop development).
2. The VLM emits one structured **observation per clip**: a verb, an
   object target, the visible ingredients and tools, and the observed states
   of objects.
3. Observations are appended to an **`ObservationHistory`** — an append-only
   list. The next window's prompt receives a short summary of earlier
   observations, so the VLM has context for what has already happened.
4. From this history we derive the current state of the kitchen on read
   (union of ingredients seen, latest state per object, and an ordered
   action-object sequence). This will feed the upcoming recipe-belief module
   that scores the history against the static recipe graphs in
   `recipe_graphs/`.

The system is intentionally append-only — earlier observations are never
overwritten — so we keep a faithful audit trail per session.

## Long-horizon reasoning fix

The current pipeline stores the right observations, but some of the derived
views are still too flat for recipe recognition over longer sequences.
`actions_so_far()` keeps the order of verbs, but it drops the object target,
so:

```text
["chop", "saute", "add", "saute"]
```

is less useful than:

```text
[("chop", "onion"), ("saute", "onion"), ("add", "minced meat"), ("saute", "minced meat")]
```

That action-object trajectory is a better fit for future recipe matching,
because recipes depend on what happened, to what, and in what order.

To improve this without changing the architecture, I added
`ObservationHistory.action_sequence()`. It derives an ordered list of
`(verb, object_target)` pairs directly from the append-only history. Nothing
about storage, parsing, prompting, or JSON output changed. This is just a
new read-only view over the same observations.

This is intentionally the first step rather than full graph reasoning. It is
simple, easy to test, and immediately useful for future recipe-belief scoring.

Possible next extensions from the same history:

- state histories, e.g. `onion: raw -> diced -> browning`
- object lifecycles, such as first appearance, transformations, and
  interactions
- recipe graph matching between the observed action sequence and a static
  recipe graph

## Prompt scalability fix

The scalability issue was in prompt generation, not in
`ObservationHistory`. Originally, `build_prompt()` replayed every past
observation into each new VLM call, so the prompt kept growing as the session
got longer.

I kept the append-only history exactly as it was and changed only how that
history is rendered for prompting. In `src/dynamic_graph.py` I added
`ObservationHistory.compact_context(recent_windows=5)` and updated
`build_prompt()` to use it.

The compact context includes:

- kitchen state from `ingredients_seen()`, `tools_seen()`, and `latest_states()`
- recent actions only
- the last few observations only

This keeps the full history intact for logging, debugging, and future
recipe-belief inference, while keeping the prompt much more stable over long
videos.

Before, the history section looked like this:

```text
window 1: verb=chop, target=onion, ingredients=[onion], tools=[knife, cutting board]
window 2: verb=saute, target=onion, ingredients=[onion, oil], tools=[pan, spatula]
window 3: verb=add, target=minced meat, ingredients=[minced meat], tools=[pan]
...
```

After, it looks more like this:

```text
Kitchen state:
* ingredients_seen: onion, oil, minced meat, salt, tomato sauce
* tools_seen: knife, cutting board, pan, spatula
* latest_states: onion=browning, minced meat=browning, sauce=mixed

Recent actions:
* saute onion
* add minced meat
* stir meat mixture
* season meat mixture
* pour tomato sauce

Recent observations (last 5 windows):
* window 2: Person stirs onions in a hot pan.
* window 3: Person adds minced meat to the pan.
* window 4: Person stirs the meat and onions.
* window 5: Person seasons the mixture.
* window 6: Person pours tomato sauce into the pan.
```

I did not change `WindowObservation`, `parse_observation()`,
`run_dynamic_loop.py`, the JSON output format, or how observations are stored.
The history is still fully preserved; only the prompt summary is now more
compact.

The old prompt grew with the **total number of windows**. The new prompt is
bounded by:

- the number of distinct ingredients/tools/states seen so far
- the number of recent windows included in context

So the cost no longer grows with the full session length in the same way. The
history remains complete in memory and in JSON logs, but the VLM only receives
the compact context it needs for the next step.

## Repository layout

```
rssp-group1/
├── dataset.csv                    Recipes selected for high overlap (peer)
├── generate_graphs_instruct.py    Static recipe-graph generation via LM Studio (peer)
├── generate_graphs_vl.py          Variant of the above (peer)
├── extract_frames.py              Frame-based video preprocessing (peer)
├── vlm_probe.py                   Frame-based VLM probe via LM Studio (peer)
├── recipe_graphs/
│   ├── instruct/                  Hand-curated static recipe graphs (peer)
│   └── vl/                        VL-prompted variants (peer)
├── src/
│   ├── dynamic_graph.py           NEW · per-window observation records + prompts
│   ├── run_dynamic_loop.py        NEW · run the observation loop across a list of clips
│   ├── vlm_baseline.py            Earlier whole-video baseline
│   ├── vlm_image_test.py          Minimal single-image sanity check
│   ├── prompts.py                 Prompt templates for the baseline
│   └── video_utils.py             Clip extraction helpers
├── data/                          Videos go here (git-ignored)
├── outputs/                       Pipeline outputs (git-ignored)
└── results/                       Earlier results (peer)
```

## Files added or substantially changed in this branch

| File | Status | Purpose |
|---|---|---|
| `src/dynamic_graph.py` | **new** | Defines `WindowObservation` (one structured record per clip) and `ObservationHistory` (append-only list across clips). Includes the prompt builder, the VLM-response parser, and a self-contained smoke test. |
| `src/run_dynamic_loop.py` | **new** | Driver that runs the observation loop end-to-end: loads Qwen2.5-VL, processes each clip in order, appends to the history, prints and logs per-window results. |
| `src/vlm_baseline.py` | adjusted | Default model switched to `Qwen2.5-VL-3B-Instruct` and load path simplified to use MPS / CUDA / CPU explicitly (no `device_map="auto"`). |
| `src/vlm_image_test.py` | new | Minimum-viable image-only test (no video pipeline). Useful for verifying the model loads and produces output before debugging the video path. |
| `requirements.txt` | adjusted | Trimmed to the minimum for Mac / Linux: no `bitsandbytes`, no `decord`. |

## Setup

```bash
git clone <repo-url>
cd rssp-group1

# Python 3.10+ recommended (3.9 works but yt-dlp & some hints are deprecated)
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Hardware notes:

- **CUDA GPU with ≥ 24 GB VRAM** runs the 7B model comfortably.
- **Apple Silicon (M2 Pro and up, 24 GB unified memory)** runs the 3B model
  on MPS; the 7B works but is slow. Set the env var
  `PYTORCH_ENABLE_MPS_FALLBACK=1` before launching to silence MPS-unsupported
  op errors.

Videos are not committed. Put them in `data/`.

## Running the per-window observation loop

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.run_dynamic_loop \
    --clips data/chopping_onions.mp4 \
            data/cooking_onions.mp4 \
            data/cooking_minced_meat.mp4 \
    --recipe "Spaghetti bolognese early steps" \
    --fps 0.5
```

For each clip the driver prints the VLM's raw response, the parsed
observation, and the history after this window. Per-window JSON logs are
written to `outputs/dynamic_loop/window_NN.json`. The final accumulated
history is written to `outputs/dynamic_loop/final_history.json`.

To verify the observation logic without loading the model, run the smoke test:

```bash
python src/dynamic_graph.py
```

This applies three hand-written VLM responses to an `ObservationHistory` and
prints the resulting structure plus the prompt that would be sent next.

## Other entry points (older / simpler)

- `src/vlm_baseline.py` — single VLM call over a whole video, with one
  free-form prompt. Predates the observation-loop design; useful for one-off
  sanity checks.
- `src/vlm_image_test.py` — minimum-viable image call, for verifying the
  model is loaded and producing output before involving the video pipeline.

## Next steps

- Score the accumulated `ObservationHistory` against each static recipe
  graph in `recipe_graphs/instruct/` to produce a probability distribution
  over candidate recipes.
- Entropy-based trigger that fires a clarification question when the belief
  is too uncertain.
- Two clarification-prompt templates (open wh vs polar restricted-offer)
  for the form-of-question comparison described in the proposal.
- Evaluation script that computes information gain per clarification turn.
