"""Baseline: run Qwen2.5-VL on a cooking video and print its description.

This is intentionally the simplest useful pipeline. It:
  1. Optionally extracts a short clip from a longer source video.
  2. Loads Qwen2.5-VL-3B-Instruct (or another variant).
  3. Asks the VLM a single prompt about the clip.
  4. Saves the model's response to a JSON file.

Run from the repo root:

    python -m src.vlm_baseline --video data/sample.mp4 --prompt-name describe

For a long HD-EPIC video, take a 15 s window starting at minute 1:

    python -m src.vlm_baseline \
        --video data/hd-epic/P01/session-01.mp4 \
        --start-sec 60 --duration-sec 15 \
        --prompt-name recipe_guess \
        --output outputs/p01_60s_recipe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# local
from .prompts import get_prompt, PROMPTS
from .video_utils import extract_clip, get_video_duration


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
# For laptop / small-GPU development you can swap this to:
#     "Qwen/Qwen2.5-VL-3B-Instruct"


def load_model(model_name: str = DEFAULT_MODEL):
    """Load Qwen2.5-VL model and its processor."""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[vlm_baseline] loading {model_name} on {device} ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def run_on_video(
    model,
    processor,
    video_path: str | Path,
    prompt_text: str,
    max_new_tokens: int = 512,
    fps: float = 1.0,
    max_pixels: int = 360 * 420,
) -> str:
    """Run the model on `video_path` with the given text prompt.

    `fps` and `max_pixels` are passed to qwen-vl-utils. Lower them to reduce
    token cost (and run time) at the price of fidelity.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "max_pixels": max_pixels,
                    "fps": fps,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="path to source video (mp4)")
    p.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="if given, extract a clip starting at this second",
    )
    p.add_argument(
        "--duration-sec",
        type=float,
        default=15.0,
        help="duration of the extracted clip (used only if --start-sec is set)",
    )
    p.add_argument(
        "--prompt-name",
        default="describe",
        choices=list(PROMPTS.keys()),
        help="which prompt template to use (see src/prompts.py)",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model id (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="frames per second to sample for the VLM (lower = cheaper)",
    )
    p.add_argument(
        "--max-pixels",
        type=int,
        default=360 * 420,
        help="max pixel budget per frame (lower = cheaper)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="maximum tokens the model may generate",
    )
    p.add_argument(
        "--output",
        default=None,
        help="path to write the JSON output (default: outputs/<videoname>_<prompt>.json)",
    )
    args = p.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[vlm_baseline] ERROR: video not found: {video_path}", file=sys.stderr)
        return 2

    # Optionally extract a clip
    if args.start_sec is not None:
        total = get_video_duration(video_path)
        if args.start_sec + args.duration_sec > total:
            print(
                f"[vlm_baseline] WARNING: clip window "
                f"[{args.start_sec}, {args.start_sec + args.duration_sec}) "
                f"goes past the end of the video ({total:.1f} s).",
                file=sys.stderr,
            )
        clip_name = (
            f"{video_path.stem}_clip_{int(args.start_sec)}s_"
            f"{int(args.duration_sec)}s.mp4"
        )
        clip_path = Path("outputs") / clip_name
        print(f"[vlm_baseline] extracting clip to {clip_path}", flush=True)
        extract_clip(video_path, args.start_sec, args.duration_sec, clip_path)
        video_for_model = clip_path
    else:
        video_for_model = video_path

    # Load model
    t0 = time.time()
    model, processor = load_model(args.model)
    print(f"[vlm_baseline] model loaded in {time.time() - t0:.1f} s", flush=True)

    # Run
    prompt_text = get_prompt(args.prompt_name)
    print(f"[vlm_baseline] running on {video_for_model} with prompt {args.prompt_name!r}...", flush=True)
    t0 = time.time()
    response = run_on_video(
        model,
        processor,
        video_path=video_for_model,
        prompt_text=prompt_text,
        max_new_tokens=args.max_new_tokens,
        fps=args.fps,
        max_pixels=args.max_pixels,
    )
    elapsed = time.time() - t0
    print(f"[vlm_baseline] generation took {elapsed:.1f} s", flush=True)

    # Save
    out_path = (
        Path(args.output)
        if args.output
        else Path("outputs") / f"{video_for_model.stem}_{args.prompt_name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "video": str(video_for_model),
        "source_video": str(video_path),
        "start_sec": args.start_sec,
        "duration_sec": args.duration_sec if args.start_sec is not None else None,
        "prompt_name": args.prompt_name,
        "prompt_text": prompt_text,
        "fps": args.fps,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": elapsed,
        "response": response,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[vlm_baseline] wrote {out_path}", flush=True)

    print("\n----- VLM response -----\n")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
