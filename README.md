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
   (union of ingredients seen, latest state per object, action sequence so
   far). This will feed the upcoming recipe-belief module that scores the
   history against the static recipe graphs in `recipe_graphs/`.

The system is intentionally append-only — earlier observations are never
overwritten — so we keep a faithful audit trail per session.

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
