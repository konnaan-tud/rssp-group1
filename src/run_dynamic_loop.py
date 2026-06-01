"""Run the per-window observation loop across a sequence of video clips.

For each clip the VLM emits one structured observation (verb, object_target,
ingredients, tools, object_states). We append each observation to an
ObservationHistory and pass the running history into the next prompt so the
model has context for what it has already seen.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.run_dynamic_loop \\
        --clips data/clip_step1.mp4 data/clip_step2.mp4 data/clip_step3.mp4 \\
        --recipe "Cucumber salad" \\
        --fps 0.5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from .dynamic_graph import (
    ObservationHistory, build_prompt, parse_observation,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str = DEFAULT_MODEL):
    device = pick_device()
    print(f"[loop] loading {model_name} on {device} ...", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"[loop] model loaded in {time.time() - t0:.1f} s", flush=True)
    return model, processor, device


def run_single_window(model, processor, device, history, window_idx, video_path,
                      recipe_summary, fps=0.5, max_new_tokens=512):
    """One VLM call. Returns (parsed_observation, raw_response_text)."""
    prompt = build_prompt(history, recipe_summary)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path,
                 "fps": fps, "max_pixels": 360 * 420},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    raw = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0]

    obs = parse_observation(raw, window=window_idx, clip_path=video_path)
    return obs, raw


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", nargs="+", required=True,
                   help="ordered list of video clip paths (one per observation window)")
    p.add_argument("--recipe", default="Unknown cooking activity.",
                   help="short text describing what the system should assume")
    p.add_argument("--fps", type=float, default=0.5)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--log-dir", default="outputs/dynamic_loop",
                   help="where to write per-clip JSON logs")
    args = p.parse_args()

    missing = [c for c in args.clips if not Path(c).exists()]
    if missing:
        print(f"[loop] ERROR: clips not found: {missing}", file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    history = ObservationHistory()
    model, processor, device = load_model(args.model)

    for i, clip_path in enumerate(args.clips, start=1):
        print(f"\n========== Window {i}/{len(args.clips)}: {clip_path} ==========",
              flush=True)
        t0 = time.time()
        obs, raw = run_single_window(
            model, processor, device, history,
            window_idx=i,
            video_path=clip_path,
            recipe_summary=args.recipe,
            fps=args.fps,
        )
        history.add(obs)
        elapsed = time.time() - t0

        print(f"[loop] window {i} took {elapsed:.1f} s", flush=True)
        print("\n--- VLM raw response ---")
        print(raw)
        print("\n--- Parsed observation ---")
        print(json.dumps(obs.to_dict(), indent=2))
        print("\n--- History so far ---")
        print(history.to_json())

        log_path = log_dir / f"window_{i:02d}.json"
        log_path.write_text(json.dumps({
            "window": i,
            "clip": clip_path,
            "elapsed_seconds": elapsed,
            "raw_response": raw,
            "parsed_observation": obs.to_dict(),
            "history_after": history.to_dict(),
        }, indent=2))
        print(f"[loop] wrote {log_path}", flush=True)

    final_path = log_dir / "final_history.json"
    final_path.write_text(history.to_json())
    print(f"\n[loop] final history written to {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
