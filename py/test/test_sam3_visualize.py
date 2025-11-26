"""
Test SAM3 streaming video inference and visualize detected objects with masks.

This test uses multiple prompts to detect objects and draws their masks
on the first frame, saving to output.jpg.
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from accelerate import Accelerator

# Setup device
device = Accelerator().device
print(f"Using device: {device}")

# Enable cudnn autotuner for repeated inference
torch.backends.cudnn.benchmark = True

def apply_mask_with_color(image, mask, color, alpha=0.5):
    """Apply a colored mask overlay to an image."""
    # Convert mask to numpy if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Ensure mask is 2D
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    # Create colored overlay
    overlay = image.copy()
    overlay_array = np.array(overlay)
    
    # Apply color where mask is True
    mask_bool = mask > 0.5
    for c in range(3):
        overlay_array[:, :, c] = np.where(
            mask_bool,
            overlay_array[:, :, c] * (1 - alpha) + color[c] * alpha,
            overlay_array[:, :, c]
        )
    
    return Image.fromarray(overlay_array.astype(np.uint8))

def draw_box_and_label(draw, box, label, color, font=None):
    """Draw bounding box and label on image."""
    x1, y1, x2, y2 = [int(coord) for coord in box]
    
    # Draw box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # Draw label background
    if font:
        bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1), label, fill='white', font=font)
    else:
        # Fallback without font
        text_bbox = draw.textbbox((x1, y1 - 20), label)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
        draw.text((x1, y1 - 20), label, fill='white')

def visualize_detections(frame, outputs, text_prompts):
    """Visualize detected objects with masks, boxes, and labels."""
    # Convert frame to PIL Image if needed
    if isinstance(frame, np.ndarray):
        frame_pil = Image.fromarray(frame)
    elif isinstance(frame, torch.Tensor):
        frame_pil = Image.fromarray(frame.cpu().numpy())
    else:
        frame_pil = frame
    
    # Ensure RGB mode
    if frame_pil.mode != 'RGB':
        frame_pil = frame_pil.convert('RGB')
    
    # Define colors for each object (distinct colors)
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
    ]
    
    # Start with original image
    result = frame_pil.copy()
    
    # Get detections
    object_ids = outputs['object_ids'].tolist()
    scores = outputs['scores'].tolist()
    boxes = outputs['boxes']
    masks = outputs['masks']
    
    print(f"\nVisualizing {len(object_ids)} detected objects...")
    
    # Apply masks with different colors
    for i, (obj_id, score) in enumerate(zip(object_ids, scores)):
        color = colors[i % len(colors)]
        mask = masks[i]
        
        # Resize mask to match frame dimensions if needed
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        
        # Apply colored mask
        result = apply_mask_with_color(result, mask_np, color, alpha=0.4)
        
        print(f"  Object {i}: ID={obj_id}, Score={score:.3f}, Color=RGB{color}")
    
    # Draw boxes and labels
    draw = ImageDraw.Draw(result)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = None
    
    for i, (obj_id, score, box) in enumerate(zip(object_ids, scores, boxes)):
        color = colors[i % len(colors)]
        label = f"ID{obj_id}: {score:.2f}"
        draw_box_and_label(draw, box.tolist(), label, color, font)
    
    return result

def main():
    print("="*70)
    print("SAM3 Multi-Object Detection Visualization")
    print("="*70)
    
    # Load model and processor
    print("\nLoading model and processor...")
    load_start = time.time()
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    
    # Load video
    print("\nLoading video...")
    video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
    video_frames, _ = load_video(video_url)
    print(f"Video loaded: {len(video_frames)} frames")
    print(f"Frame shape: {video_frames[0].size if hasattr(video_frames[0], 'size') else video_frames[0].shape}")
    
    # Text prompts for detection
    text_prompts = ["person", "cat", "dog", "chair", "bed", "lamp"]
    print(f"\nText prompts: {text_prompts}")
    
    # Initialize streaming session
    print("\nInitializing streaming session...")
    stream_session = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    
    # Add all text prompts
    print(f"Adding {len(text_prompts)} text prompts...")
    for prompt in text_prompts:
        print(f"  Adding prompt: '{prompt}'")
        stream_session = processor.add_text_prompt(
            inference_session=stream_session,
            text=prompt,
        )
    
    # Process first frame only
    print("\nProcessing first frame...")
    frame = video_frames[0]
    
    # Process frame
    inputs = processor(images=frame, device=device, return_tensors="pt")
    
    # Run inference
    model_outputs = model(
        inference_session=stream_session,
        frame=inputs.pixel_values[0],
        reverse=False,
    )
    
    # Post-process outputs
    processed_outputs = processor.postprocess_outputs(
        stream_session,
        model_outputs,
        original_sizes=inputs.original_sizes,
    )
    
    # Print detection results
    print("\n" + "="*70)
    print("DETECTION RESULTS:")
    print("="*70)
    print(f"Objects detected: {len(processed_outputs['object_ids'])}")
    print(f"Object IDs: {processed_outputs['object_ids'].tolist()}")
    print(f"Scores: {[f'{s:.3f}' for s in processed_outputs['scores'].tolist()]}")
    print(f"Boxes: {processed_outputs['boxes'].tolist()}")
    print(f"Masks shape: {processed_outputs['masks'].shape}")
    
    # Visualize results
    print("\n" + "="*70)
    print("CREATING VISUALIZATION:")
    print("="*70)
    result_image = visualize_detections(frame, processed_outputs, text_prompts)
    
    # Save output
    output_path = Path(__file__).parent / "output.jpg"
    result_image.save(output_path, quality=95)
    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"  Image size: {result_image.size}")
    
    # Also save without masks (just boxes)
    output_boxes_path = Path(__file__).parent / "output_boxes.jpg"
    frame_with_boxes = frame.copy() if hasattr(frame, 'copy') else Image.fromarray(np.array(frame))
    if not isinstance(frame_with_boxes, Image.Image):
        frame_with_boxes = Image.fromarray(frame_with_boxes)
    
    draw = ImageDraw.Draw(frame_with_boxes)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = None
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i, (obj_id, score, box) in enumerate(zip(
        processed_outputs['object_ids'].tolist(),
        processed_outputs['scores'].tolist(),
        processed_outputs['boxes']
    )):
        color = colors[i % len(colors)]
        label = f"ID{obj_id}: {score:.2f}"
        draw_box_and_label(draw, box.tolist(), label, color, font)
    
    frame_with_boxes.save(output_boxes_path, quality=95)
    print(f"✓ Boxes-only saved to: {output_boxes_path}")
    
    print("\n" + "="*70)
    print("✓ Test completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
