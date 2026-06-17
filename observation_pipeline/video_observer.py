"""
observation_pipeline/video_observer.py
--------------------------------------
Ollama-backed version. Replaces the HuggingFace transformers loader with
calls to a local Ollama instance running qwen2.5vl:7b.

Video observation:
  - Frames are extracted from each clip with PyAV at DEFAULT_FPS.
  - Frames are JPEG-encoded and base64-encoded, then sent to Ollama's
    /api/chat endpoint as images in the message payload.
  - Ollama handles quantisation internally (~6GB VRAM vs ~15GB for float16).

Function signatures are identical to the transformers version so the
orchestrators require no changes:
  load_qwen_vlm()          → (model_name, None, "ollama")
  describe_clip(path, ...) → one scene sentence string
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import av
import requests
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434"   # default Ollama address
OLLAMA_MODEL = "qwen2.5vl:7b"

DEFAULT_FPS        = 1.0    # frames per second sampled from each clip
MAX_FRAMES         = 8      # hard cap — keeps payloads manageable
MAX_NEW_TOKENS_OBS = 64     # short cap; scene sentences are brief


# ─────────────────────────────────────────────────────────────────────────
# Observation prompt — unchanged from transformers version
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
# Ollama connectivity check
# ─────────────────────────────────────────────────────────────────────────

def load_qwen_vlm(model_name: str = OLLAMA_MODEL):
    """
    For Ollama, there is no model to load into memory — Ollama manages
    the model as a server-side process. This function verifies that Ollama
    is reachable and that the requested model is available, then returns
    (model_name, None, "ollama") to keep the orchestrator call-sites intact.

    Parameters
    ----------
    model_name : Ollama model tag, e.g. "qwen2.5vl:7b"
    """
    print(f"[VLM] Checking Ollama at {OLLAMA_URL} ...", flush=True)

    # Ping Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Ollama is not running. Start it with: ollama serve\n"
            f"  (expected at {OLLAMA_URL})"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama did not respond within 5 s at {OLLAMA_URL}.")

    # Check the model is pulled
    available = [m["name"] for m in r.json().get("models", [])]
    # Ollama may list the model with or without the tag suffix
    base = model_name.split(":")[0]
    if not any(base in name for name in available):
        raise RuntimeError(
            f"Model '{model_name}' not found in Ollama.\n"
            f"  Pull it with: ollama pull {model_name}\n"
            f"  Available models: {available}"
        )

    print(f"[VLM] Ollama ready — model: {model_name}", flush=True)
    # Return (model_name, None, "ollama") to keep orchestrator signatures intact.
    # Callers receive these three values and pass them to describe_clip /
    # generate_text_with_model, which detect device == "ollama" and route
    # to the Ollama API instead of a local torch model.
    return model_name, None, "ollama"


# ─────────────────────────────────────────────────────────────────────────
# Frame extraction
# ─────────────────────────────────────────────────────────────────────────

def _extract_frames_b64(
    video_path: str | Path,
    fps: float = DEFAULT_FPS,
    max_frames: int = MAX_FRAMES,
) -> list[str]:
    """
    Extract up to `max_frames` frames from `video_path` at `fps` rate.
    Returns a list of base64-encoded JPEG strings ready for the Ollama
    image payload. Returns [] on any read error.
    """
    frames_b64: list[str] = []

    try:
        container = av.open(str(video_path))
    except Exception as e:
        print(f"  [video] Could not open {video_path}: {e}")
        return []

    try:
        video_stream = container.streams.video[0]
        video_stream.thread_type = "AUTO"

        interval   = 1.0 / max(fps, 0.1)
        last_t     = -interval   # ensure first frame is always captured

        for frame in container.decode(video=0):
            if len(frames_b64) >= max_frames:
                break

            # Compute timestamp in seconds
            if frame.pts is not None and frame.time_base is not None:
                t = float(frame.pts * frame.time_base)
            else:
                t = last_t + interval   # fallback: accept the frame

            if t - last_t < interval:
                continue
            last_t = t

            # Convert to PIL, downscale, JPEG-encode, base64-encode
            img = frame.to_image().convert("RGB")
            img.thumbnail((640, 480), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    except Exception as e:
        print(f"  [video] Frame extraction error: {e}")
    finally:
        container.close()

    return frames_b64


# ─────────────────────────────────────────────────────────────────────────
# Video observation via Ollama
# ─────────────────────────────────────────────────────────────────────────

def describe_clip(
    video_path: str | Path,
    model,            # model_name string returned by load_qwen_vlm
    processor,        # unused (None) — kept for signature compatibility
    device: str,      # "ollama" — kept for signature compatibility
    fps: float = DEFAULT_FPS,
    max_new_tokens: int = MAX_NEW_TOKENS_OBS,
) -> str:
    """
    Describe a cooking clip in one sentence via Ollama.

    Extracts frames with PyAV, sends them as images to Ollama's /api/chat
    endpoint, and returns the cleaned scene sentence.
    """
    frames_b64 = _extract_frames_b64(video_path, fps)
    if not frames_b64:
        print(f"  [video] No frames extracted from {video_path} — skipping.")
        return ""

    print(f"  [video] Sending {len(frames_b64)} frame(s) to Ollama...", flush=True)

    payload = {
        "model":   model,
        "messages": [
            {
                "role":    "user",
                "content": OBSERVATION_PROMPT,
                "images":  frames_b64,
            }
        ],
        "stream":  False,
        "options": {"num_predict": max_new_tokens},
    }

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
    except requests.exceptions.Timeout:
        print("  [video] Ollama request timed out.")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"  [video] Ollama request failed: {e}")
        return ""

    raw = r.json().get("message", {}).get("content", "")
    return _clean_sentence(raw)


# ─────────────────────────────────────────────────────────────────────────
# Output cleaning — unchanged
# ─────────────────────────────────────────────────────────────────────────

def _clean_sentence(raw: str) -> str:
    """
    Tidy the VLM output:
      - Strip whitespace / quotes
      - Keep only the first sentence (split on '.', '?', '!')
      - Ensure it ends with a period
    """
    s = raw.strip().strip('"').strip("'").strip()

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