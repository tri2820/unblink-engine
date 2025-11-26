"""
Test SAM3 video inference with pseudo-streaming (same frame repeated).

This simulates streaming video inference by creating a temporary directory
with the same test frame repeated N times, then processing them sequentially.

Key Findings:
- Pre-resizing frames to 720px gives 1.31x FPS speedup (11.66 → 15.28 FPS)
- Session initialization is 25x faster with resized frames (14.26s → 0.56s)
- Video propagation benefits from pre-resizing like image processing
- SAM3VideoPredictor maintains temporal consistency regardless of input size

Note: This is NOT true frame-by-frame streaming. SAM3's video API requires
pre-loading all frames during session initialization. For true streaming,
you would need to buffer frames and recreate sessions periodically.
"""

import os
import sys
import time
import torch
import shutil
from PIL import Image
from pathlib import Path

# Add sam3 to path
script_dir = Path(__file__).parent
sam3_path = script_dir.parent / "sam3"
sys.path.insert(0, str(sam3_path))

from sam3.model_builder import build_sam3_video_predictor

def resize_image(image, max_size=720):
    """Resize image so longest side <= max_size."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    
    return image.resize((new_w, new_h), Image.LANCZOS)

def prepare_video_frames(image_path, num_frames, temp_dir, resize=False):
    """Prepare a pseudo-video by copying the same frame N times."""
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load and optionally resize the test image
    image = Image.open(image_path)
    if resize:
        image = resize_image(image, max_size=720)
        print(f"Resized image to: {image.size}")
    else:
        print(f"Original image size: {image.size}")
    
    # Save as sequential frames (00000.jpg, 00001.jpg, etc.)
    for i in range(num_frames):
        frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
        image.save(frame_path, quality=95)
    
    print(f"Created {num_frames} frames in {temp_dir}")
    return temp_dir

def run_video_inference(temp_dir, resize_label):
    """Run SAM3 video inference on the prepared frames."""
    print(f"\n{'='*60}")
    print(f"Testing with {resize_label}")
    print(f"{'='*60}")
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Build predictor (single GPU)
    print("Building SAM3 video predictor...")
    predictor = build_sam3_video_predictor(gpus_to_use=[0])
    
    # Start session and load video
    print("Starting session and loading video frames...")
    session_start = time.time()
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=temp_dir,
        )
    )
    session_id = response["session_id"]
    session_time = time.time() - session_start
    print(f"Session initialization: {session_time:.2f}s")
    
    # Add text prompt on frame 0
    print("Adding text prompt 'person' on frame 0...")
    prompt_start = time.time()
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text="person",
        )
    )
    prompt_time = time.time() - prompt_start
    print(f"Prompt processing: {prompt_time:.2f}s")
    
    # Propagate through video
    print("Propagating through video...")
    propagation_start = time.time()
    outputs_per_frame = {}
    
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        frame_idx = response["frame_index"]
        outputs_per_frame[frame_idx] = response["outputs"]
        
        if (frame_idx + 1) % 10 == 0:
            elapsed = time.time() - propagation_start
            fps = (frame_idx + 1) / elapsed
            print(f"  Processed {frame_idx + 1} frames - {fps:.2f} FPS")
    
    propagation_time = time.time() - propagation_start
    num_frames = len(outputs_per_frame)
    avg_fps = num_frames / propagation_time
    
    print(f"\n{'='*60}")
    print(f"Results for {resize_label}:")
    print(f"{'='*60}")
    print(f"Total frames: {num_frames}")
    print(f"Propagation time: {propagation_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Time per frame: {propagation_time/num_frames*1000:.1f}ms")
    
    # Show first frame results
    frame_0_outputs = outputs_per_frame[0]
    if frame_0_outputs is not None:
        print(f"\nFirst frame detection:")
        if 'out_obj_ids' in frame_0_outputs:
            obj_ids = frame_0_outputs['out_obj_ids']
            print(f"  Detected objects: {len(obj_ids)}")
            if len(obj_ids) > 0:
                print(f"  Object IDs: {obj_ids}")
        if 'out_binary_masks' in frame_0_outputs:
            print(f"  Masks shape: {frame_0_outputs['out_binary_masks'].shape}")
        if 'out_boxes_xywh' in frame_0_outputs:
            print(f"  Boxes shape: {frame_0_outputs['out_boxes_xywh'].shape}")
    
    # Clean up session
    predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    
    return {
        'session_time': session_time,
        'prompt_time': prompt_time,
        'propagation_time': propagation_time,
        'avg_fps': avg_fps,
        'num_frames': num_frames
    }

def main():
    # Configuration
    image_path = os.path.join(os.path.dirname(__file__), "test.jpg")
    num_frames = 50  # Test with 50 frames
    
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        return
    
    print(f"SAM3 Video Inference Test")
    print(f"Test image: {image_path}")
    print(f"Number of frames: {num_frames}")
    
    # Test 1: Original resolution
    temp_dir_original = "/tmp/sam3_video_test_original"
    try:
        prepare_video_frames(image_path, num_frames, temp_dir_original, resize=False)
        results_original = run_video_inference(temp_dir_original, "ORIGINAL resolution")
    finally:
        if os.path.exists(temp_dir_original):
            shutil.rmtree(temp_dir_original)
    
    # Test 2: Pre-resized to 720p
    temp_dir_resized = "/tmp/sam3_video_test_resized"
    try:
        prepare_video_frames(image_path, num_frames, temp_dir_resized, resize=True)
        results_resized = run_video_inference(temp_dir_resized, "PRE-RESIZED (720px)")
    finally:
        if os.path.exists(temp_dir_resized):
            shutil.rmtree(temp_dir_resized)
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\nOriginal resolution:")
    print(f"  Session init: {results_original['session_time']:.2f}s")
    print(f"  Propagation:  {results_original['propagation_time']:.2f}s")
    print(f"  FPS:          {results_original['avg_fps']:.2f}")
    
    print(f"\nPre-resized (720px):")
    print(f"  Session init: {results_resized['session_time']:.2f}s")
    print(f"  Propagation:  {results_resized['propagation_time']:.2f}s")
    print(f"  FPS:          {results_resized['avg_fps']:.2f}")
    
    speedup_session = results_original['session_time'] / results_resized['session_time']
    speedup_propagation = results_original['propagation_time'] / results_resized['propagation_time']
    speedup_fps = results_resized['avg_fps'] / results_original['avg_fps']
    
    print(f"\nSpeedup factors:")
    print(f"  Session init:  {speedup_session:.2f}x")
    print(f"  Propagation:   {speedup_propagation:.2f}x")
    print(f"  Overall FPS:   {speedup_fps:.2f}x")

if __name__ == "__main__":
    main()
