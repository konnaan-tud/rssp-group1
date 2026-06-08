"""Run one dynamic-graph update on a video clip.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.run_dynamic_update \\
        --video data/clip_step1.mp4 \\
        --recipe "Spaghetti bolognese"
"""

import argparse
import sys
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from .dynamic_graph import DynamicSceneGraph, build_update_prompt, parse_vlm_delta


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str = DEFAULT_MODEL):
    device = pick_device()
    print(f"[dyn] loading {model_name} on {device} ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor, device


def update_graph_from_video(
    graph: DynamicSceneGraph,
    model,
    processor,
    device: str,
    video_path: str,
    recipe_summary: str,
    fps: float = 0.5,
    max_new_tokens: int = 512,
) -> tuple[dict, str]:
    """Send (current graph, video clip) to the VLM. Return (parsed_delta, raw_response)."""
    prompt = build_update_prompt(graph, recipe_summary)
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

    delta = parse_vlm_delta(raw)
    graph.apply_delta(delta)
    return delta, raw


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="video clip to process")
    p.add_argument("--recipe", default="Unknown.",
                   help="short text describing what recipe context the VLM should assume")
    p.add_argument("--fps", type=float, default=0.5)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    graph = DynamicSceneGraph()
    model, processor, device = load_model(args.model)

    delta, raw = update_graph_from_video(
        graph, model, processor, device,
        video_path=args.video,
        recipe_summary=args.recipe,
        fps=args.fps,
    )

    print("\n----- VLM raw response -----\n")
    print(raw)
    print("\n----- Parsed delta -----\n")
    import json
    print(json.dumps(delta, indent=2))
    print("\n----- Resulting graph -----\n")
    print(graph.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
