# Project 1 — Cooking Intent Recognition with VLM Clarification

Baseline pipeline for a VLM-based cooking observer that watches a human prepare
a meal and infers the recipe being prepared. This repository hosts the initial
infrastructure; the methodology, comparison of clarification question forms,
and full evaluation are described in the project proposal.

## What this repo does (so far)

A minimal pipeline that:

1. Takes a cooking video (e.g. a clip from the HD-EPIC dataset).
2. Optionally extracts a short clip from a longer video at a given start time.
3. Passes the clip to **Qwen2.5-VL-7B-Instruct**.
4. Asks the VLM what is happening and what recipe might be being prepared.
5. Writes the model's response to `outputs/`.

This is **only the perception baseline**. The clarification loop, the
recipe-belief module, and the experimental comparison are not yet implemented.

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd project1-cooking-intent

# 2. Python environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Hardware notes

- The 7B model is roughly 16 GB in `bfloat16`. A single GPU with **≥ 24 GB
  VRAM** runs it comfortably. A 16 GB GPU will fit it in 8-bit, see
  `requirements.txt`.
- CPU inference is not practical for the 7B model. For laptop development we
  recommend swapping to **`Qwen/Qwen2.5-VL-3B-Instruct`** (change the model
  name in `src/vlm_baseline.py`).

### Data

HD-EPIC videos are large and licensed for research use only. They are **not**
committed to the repo. Put any video files you want to run the baseline on
into `data/`. See `data/README.md` for the expected naming convention.

## Running the baseline

```bash
# Whole-video mode (only for short clips)
python -m src.vlm_baseline \
    --video data/sample.mp4 \
    --prompt-name describe \
    --output outputs/sample_describe.json

# Clip-window mode (for long HD-EPIC videos)
python -m src.vlm_baseline \
    --video data/p01.mp4 \
    --start-sec 60 --duration-sec 15 \
    --prompt-name recipe_guess \
    --output outputs/p01_clip_recipe.json
```

Available prompts live in `src/prompts.py`. The two starter prompts are:

- `describe` — open-ended description of what is happening
- `recipe_guess` — asks the VLM for its top three recipe candidates

## Next steps (tracked in issues)

- [ ] Probabilistic belief module over a finite set of candidate recipes
- [ ] Hand-curated static scene graphs for ~5 recipes
- [ ] Entropy-based trigger for asking
- [ ] Two clarification-prompt templates (open wh / polar restricted-offer)
- [ ] Evaluation script: information gain per clarification turn

## Citing

Once the proposal is finalised, see `CITATION.cff` (TBD) for the formal cite.
