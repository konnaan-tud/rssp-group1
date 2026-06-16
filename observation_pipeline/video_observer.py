"""
observation_pipeline/video_observer.py
--------------------------------------
Runs Qwen2.5-VL on a single cooking video clip and returns a one-sentence
scene description in the format used by the recipe database.

Used by the orchestrator to convert each WINDOW (a video clip) into a
scene sentence consumable by BeliefUpdaterV3.

Design:
  - load_qwen_vlm()      → load Qwen2.5-VL model + processor once at session
                            start. Returns (model, processor, device).
  - describe_clip(path)  → run inference on one clip, return a sentence
                            like "A cook cracks eggs into a mixing bowl."

Sharing the loaded model with the questioning pipeline (which also uses
Qwen2.5-VL) avoids paying the ~30s load cost on every call.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_FPS = 1.0           # frames per second extracted from each clip
MAX_NEW_TOKENS_OBS = 64     # observation output is short — caps verbosity


# ─────────────────────────────────────────────────────────────────────────
# Observation prompt
# ─────────────────────────────────────────────────────────────────────────

OBSERVATION_PROMPT = """You are watching a short cooking video clip. In ONE present-tense sentence, describe what the cook is doing.

The sentence must start with "A cook" and follow this format exactly:
"A cook [verb]s [object] [context]."

Good examples:
- "A cook pours water into a large pot."
- "A cook cracks eggs into a mixing bowl."
- "A cook grates pecorino into the mixing bowl."
- "A cook dices pancetta on a cutting board."
- "A cook drains pasta over the sink."
- "A cook stirs the pasta in the skillet."

Rules:
- Use simple present tense (pours, cracks, dices, stirs, drains, etc.).
- Start with "A cook".
- Output ONE sentence only.
- No preamble, no commentary, no explanation.
- Keep the sentence under 15 words.
"""


# ─────────────────────────────────────────────────────────────────────────
# Model loader (shared with questioning pipeline)
# ─────────────────────────────────────────────────────────────────────────

def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_qwen_vlm(model_name: str = DEFAULT_MODEL):
    """
    Load Qwen2.5-VL once. Returns (model, processor, device).

    The same model handles both video-to-text (observation) and text-only
    (question generation), so this loader is reused by both paths.
    """
    device = _pick_device()
    print(f"[VLM] Loading {model_name} on {device}...", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    print("[VLM] Ready.", flush=True)
    return model, processor, device


# ─────────────────────────────────────────────────────────────────────────
# Video observation
# ─────────────────────────────────────────────────────────────────────────

def _sample_video_frames_av(
    video_path: str | Path,
    fps: float,
) -> np.ndarray:
    """
    Decode RGB frames with PyAV.
    """
    container = av.open(str(Path(video_path).resolve()))
    stream = container.streams.video[0]

    native_fps = float(stream.average_rate) if stream.average_rate else 0.0
    if native_fps <= 0.0:
        native_fps = float(fps) if fps > 0 else 1.0

    frame_interval = max(1, int(round(native_fps / max(fps, 1e-6))))

    frames: list[np.ndarray] = []
    for frame_idx, frame in enumerate(container.decode(stream)):
        if frame_idx % frame_interval == 0:
            frames.append(frame.to_ndarray(format="rgb24"))

    container.close()

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")

    return np.stack(frames, axis=0)

def describe_clip(
    video_path: str | Path,
    model,
    processor,
    device: str,
    fps: float = DEFAULT_FPS,
    max_new_tokens: int = MAX_NEW_TOKENS_OBS,
) -> str:
    """
    Run Qwen2.5-VL on `video_path` and return a one-sentence scene
    description. Caller is responsible for loading model + processor once
    via load_qwen_vlm().

    Parameters
    ----------
    video_path : path to an mp4 (or other ffmpeg-readable) clip
    fps        : frames per second sampled from the clip (1.0 is enough
                 for typical 5s cooking actions)
    """
    video_path = str(Path(video_path).resolve())

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps},
                {"type": "text", "text": OBSERVATION_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    video_frames = _sample_video_frames_av(video_path, fps)
    inputs = processor(
        text=[text],
        videos=[video_frames],
        fps=fps,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return _clean_sentence(response)


# ─────────────────────────────────────────────────────────────────────────
# Output cleaning
# ─────────────────────────────────────────────────────────────────────────

def _clean_sentence(raw: str) -> str:
    """
    Tidy the VLM output:
      - Strip whitespace / quotes
      - Keep only the first sentence (split on '.', '?', '!')
      - Ensure it ends with a period
    """
    s = raw.strip().strip('"').strip("'").strip()

    # Pick the first sentence
    for sep in (".", "?", "!"):
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    s = s.strip()

    if not s:
        return ""

    if not s.endswith("."):
        s = s + "."
    return s
