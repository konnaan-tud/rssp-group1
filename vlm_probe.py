import os
import base64
import json
import time
import requests
from pathlib import Path

# ── Settings ───────────────────────────────────────────────────────────────
FRAMES_DIR = "frames"
RESULTS_DIR = "results"
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "qwen2.5-vl-7b-instruct"
MAX_FRAMES = 20  # cap for long clips, set to None for no cap

# ── Prompts ────────────────────────────────────────────────────────────────
PROMPT_A = {
    "system": "You are observing a cooking scene. Look carefully at the sequence of images provided, which are frames taken from a short video clip.",
    "user": "These images are sequential frames from a cooking video. Describe what is happening."
}

PROMPT_B = {
    "system": (
        "You are a cooking-scene observer assisting a robot that needs to understand human cooking intent. "
        "You are given sequential frames from a short cooking video clip. "
        "Your job is to analyse the frames carefully and return a structured JSON description of what you observe. "
        "Do not include any text outside the JSON object. Do not add markdown formatting or code fences."
    ),
    "user": (
        "Analyse these sequential cooking frames and return only a JSON object with exactly these fields:\n"
        "- \"summary\": one sentence describing the overall action sequence\n"
        "- \"actions\": list of action phrases you observe (e.g. chopping, slicing)\n"
        "- \"ingredients\": list of ingredients visible or being used\n"
        "- \"tools\": list of kitchen tools or equipment visible\n"
        "- \"object_states\": dictionary mapping each ingredient or object to its current state "
        "(e.g. {\"tomato\": \"being sliced\", \"knife\": \"in use\"})\n"
        "- \"temporal_sequence\": a short list describing how the action progresses across the frames in order"
    )
}

# ── Helpers ────────────────────────────────────────────────────────────────
def load_frames_as_base64(clip_folder, max_frames=None, max_size=512):
    """Load frames, resize them, and return as base64 strings."""
    from PIL import Image
    import io

    frame_files = sorted([
        f for f in os.listdir(clip_folder) if f.endswith(".jpg")
    ])
    if max_frames:
        frame_files = frame_files[:max_frames]

    frames = []
    for f in frame_files:
        path = os.path.join(clip_folder, f)
        img = Image.open(path)

        # Resize keeping aspect ratio
        img.thumbnail((max_size, max_size), Image.LANCZOS)

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        frames.append(encoded)

    print(f"  Loaded {len(frames)} frames ({max_size}px max) from {clip_folder}")
    return frames


def build_message(prompt, frames_b64):
    """Build the API message payload with interleaved frames."""
    content = []
    for frame in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame}",
                "detail": "low"
            }
        })
    content.append({
        "type": "text",
        "text": prompt["user"]
    })

    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": content}
    ]


def call_vlm(messages):
    """Send request to LM Studio and return raw response text."""
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1024,
    }
    response = requests.post(LM_STUDIO_URL, json=payload)
    
    if not response.ok:
        print(f"  Status code: {response.status_code}")
        print(f"  Error detail: {response.text}")
        response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"]


def save_result(clip_name, prompt_name, response_text, elapsed):
    """Save result to results/<clip_name>_<prompt_name>.txt"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = os.path.join(RESULTS_DIR, f"{clip_name}__{prompt_name}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Clip: {clip_name}\n")
        f.write(f"Prompt: {prompt_name}\n")
        f.write(f"Time: {elapsed:.1f}s\n")
        f.write("=" * 60 + "\n")
        f.write(response_text)
    print(f"  Saved → results/{clip_name}__{prompt_name}.txt")


# ── Main ───────────────────────────────────────────────────────────────────
def probe_clip(clip_name, max_frames=MAX_FRAMES):
    """Run both prompts on a single clip."""
    clip_folder = os.path.join(FRAMES_DIR, clip_name)

    if not os.path.exists(clip_folder):
        print(f"Clip folder not found: {clip_folder}")
        return

    print(f"\n{'='*60}")
    print(f"Clip: {clip_name}")
    print(f"{'='*60}")

    frames = load_frames_as_base64(clip_folder, max_frames=max_frames)

    for prompt_name, prompt in [("prompt_A_open", PROMPT_A), ("prompt_B_structured", PROMPT_B)]:
        print(f"\n  Running {prompt_name}...")
        messages = build_message(prompt, frames)

        start = time.time()
        try:
            response = call_vlm(messages)
            elapsed = time.time() - start
            print(f"  Done in {elapsed:.1f}s")
            print(f"\n  --- Response ---\n{response}\n")
            save_result(clip_name, prompt_name, response, elapsed)
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    probe_clip("stirring", max_frames=100)