"""Absolute-minimum test: load Qwen2.5-VL, send one image, print response."""

import sys
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/frame.jpg"
PROMPT = "Describe what is happening in this image. Is the person cooking?"

print(f"Loading {MODEL} on mps ...", flush=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16
).to("mps")
processor = AutoProcessor.from_pretrained(MODEL)
print("Model ready.", flush=True)

image = Image.open(IMAGE_PATH).convert("RGB")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("mps")

print("Generating...", flush=True)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=256)
response = processor.batch_decode(
    out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
)[0]

print("\n----- VLM response -----\n")
print(response)