"""
Debug script to investigate what SAM3 provides for class labels.
"""

import os
import sys
import torch
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator

# Setup device
device = Accelerator().device
print(f"Using device: {device}")

# Load model and processor
print("Loading SAM3 model and processor...")
model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
print("Model loaded successfully.")

# Load test image
test_image = Path(__file__).parent / "test.jpg"
frame = Image.open(test_image).convert("RGB")
print(f"Loaded test image: {test_image}")

# Create session with multiple prompts
print("\nCreating session with prompts: ['person', 'cat', 'dog']")
session = processor.init_video_session(
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)

# Add multiple prompts
prompts = ['person', 'cat', 'dog']
for prompt in prompts:
    session = processor.add_text_prompt(
        inference_session=session,
        text=prompt,
    )

# Process frame
print("\nProcessing frame...")
inputs = processor(images=frame, device=device, return_tensors="pt")

model_outputs = model(
    inference_session=session,
    frame=inputs.pixel_values[0],
    reverse=False,
)

processed_outputs = processor.postprocess_outputs(
    session,
    model_outputs,
    original_sizes=inputs.original_sizes,
)

# Debug outputs
print("\n" + "="*70)
print("DEBUG: processed_outputs keys")
print("="*70)
print(processed_outputs.keys())

print("\n" + "="*70)
print("DEBUG: processed_outputs values")
print("="*70)
for key, value in processed_outputs.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        if value.numel() < 20:  # Print small tensors
            print(f"  values: {value}")
    else:
        print(f"{key}: {type(value)} = {value}")

print("\n" + "="*70)
print("DEBUG: session attributes")
print("="*70)
session_attrs = [attr for attr in dir(session) if not attr.startswith('_')]
print(f"Non-private attributes: {session_attrs}")

# Check for prompt-related attributes
if hasattr(session, 'output_dict_per_obj'):
    print(f"\noutput_dict_per_obj: {session.output_dict_per_obj}")

if hasattr(session, 'prompts'):
    print(f"\nprompts: {session.prompts}")

if hasattr(session, 'prompt_texts'):
    print(f"\nprompt_texts: {session.prompt_texts}")

if hasattr(session, 'text_prompts'):
    print(f"\ntext_prompts: {session.text_prompts}")

if hasattr(session, 'obj_to_prompt_map'):
    print(f"\nobj_to_prompt_map: {session.obj_to_prompt_map}")

if hasattr(session, 'cond_frame_outputs'):
    print(f"\ncond_frame_outputs: {session.cond_frame_outputs}")

# Check model_outputs
print("\n" + "="*70)
print("DEBUG: model_outputs keys")
print("="*70)
print(model_outputs.keys())

print("\n" + "="*70)
print("DEBUG: Checking for conditioning frame info")
print("="*70)
if hasattr(session, 'output_dict_per_obj') and session.output_dict_per_obj:
    print("\noutput_dict_per_obj structure:")
    for obj_id, obj_data in list(session.output_dict_per_obj.items())[:3]:  # First 3 objects
        print(f"\nObject ID: {obj_id}")
        print(f"  Keys: {obj_data.keys() if isinstance(obj_data, dict) else 'not a dict'}")
        if isinstance(obj_data, dict):
            for key, val in obj_data.items():
                if isinstance(val, torch.Tensor):
                    print(f"    {key}: shape={val.shape}")
                elif isinstance(val, (list, tuple)) and len(val) > 0:
                    print(f"    {key}: len={len(val)}, first={val[0]}")
                else:
                    print(f"    {key}: {val}")
