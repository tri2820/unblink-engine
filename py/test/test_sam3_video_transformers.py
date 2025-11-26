"""
Test SAM3 video inference using Transformers library.

Tests both pre-loaded video inference and streaming video inference modes
to compare performance and behavior.

Based on: https://huggingface.co/docs/transformers/model_doc/sam3
"""

import os
import sys
import time
import torch
import statistics
import numpy as np
from PIL import Image
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

def resize_image(image, max_size=720):
    """Resize image so longest side <= max_size."""
    # Handle numpy arrays
    if hasattr(image, '__array__'):
        import numpy as np
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
    
    # Handle tensors
    if isinstance(image, torch.Tensor):
        # Convert tensor to PIL for resizing
        import torchvision.transforms.functional as F
        image = F.to_pil_image(image)
    
    # If it's already a PIL image
    if not isinstance(image, Image.Image):
        # Try to convert it
        try:
            image = Image.fromarray(image)
        except:
            raise ValueError(f"Cannot convert image of type {type(image)} to PIL Image")
    
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

def test_preloaded_video_inference(video_frames, processor, model, text_prompt="person", max_frames=50, resize_frames=False):
    """
    Test pre-loaded video inference where all frames are available upfront.
    This mode can use hotstart heuristics for better quality.
    """
    print(f"\n{'='*70}")
    print(f"TEST: Pre-loaded Video Inference {'(with pre-resize)' if resize_frames else '(original size)'}")
    print(f"{'='*70}")
    
    frames = video_frames[:max_frames]
    
    # Pre-resize frames if requested
    if resize_frames:
        print(f"Pre-resizing {len(frames)} frames to 720px...")
        resize_start = time.time()
        frames = [resize_image(frame, max_size=720) for frame in frames]
        resize_time = time.time() - resize_start
        print(f"Resizing took: {resize_time:.3f}s ({resize_time/len(frames)*1000:.1f}ms per frame)")
    
    # Time session initialization
    print(f"\nInitializing video session with {len(frames)} frames...")
    init_start = time.time()
    
    inference_session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    
    init_time = time.time() - init_start
    print(f"Session initialization took: {init_time:.3f}s")
    
    # Add text prompt
    print(f"Adding text prompt: '{text_prompt}'")
    prompt_start = time.time()
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=text_prompt,
    )
    prompt_time = time.time() - prompt_start
    print(f"Text prompt processing took: {prompt_time:.3f}s")
    
    # Process all frames
    print(f"\nProcessing {len(frames)} frames...")
    process_start = time.time()
    outputs_per_frame = {}
    frame_count = 0
    
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, 
        max_frame_num_to_track=max_frames
    ):
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs
        frame_count += 1
        
        if frame_count % 10 == 0:
            elapsed = time.time() - process_start
            fps = frame_count / elapsed
            print(f"  Processed {frame_count}/{len(frames)} frames ({fps:.2f} FPS)")
    
    process_time = time.time() - process_start
    total_time = init_time + prompt_time + process_time
    avg_fps = len(frames) / process_time
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"Total frames processed: {len(outputs_per_frame)}")
    print(f"Session init time: {init_time:.3f}s")
    print(f"Text prompt time: {prompt_time:.3f}s")
    print(f"Frame processing time: {process_time:.3f}s ({process_time/len(frames)*1000:.1f}ms per frame)")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    
    # Show results for first frame
    if 0 in outputs_per_frame:
        frame_0 = outputs_per_frame[0]
        print(f"\nFirst frame results:")
        print(f"  Objects detected: {len(frame_0['object_ids'])}")
        print(f"  Object IDs: {frame_0['object_ids'].tolist()}")
        print(f"  Scores: {[f'{s:.3f}' for s in frame_0['scores'].tolist()]}")
        print(f"  Boxes shape (XYXY absolute coords): {frame_0['boxes'].shape}")
        print(f"  Masks shape: {frame_0['masks'].shape}")
    
    return {
        'outputs': outputs_per_frame,
        'init_time': init_time,
        'prompt_time': prompt_time,
        'process_time': process_time,
        'total_time': total_time,
        'fps': avg_fps,
        'num_frames': len(outputs_per_frame)
    }

def test_streaming_video_inference(video_frames, processor, model, text_prompt="person", max_frames=50, resize_frames=False):
    """
    Test streaming video inference where frames arrive one at a time.
    Note: Streaming mode disables hotstart heuristics, so quality may be lower.
    """
    print(f"\n{'='*70}")
    print(f"TEST: Streaming Video Inference {'(with pre-resize)' if resize_frames else '(original size)'}")
    print(f"{'='*70}")
    print("⚠️  Note: Streaming inference may have more false positives and duplicates")
    print("    (hotstart heuristics disabled - requires future frames)")
    
    frames = video_frames[:max_frames]
    
    # Initialize session for streaming (no video provided)
    print(f"\nInitializing streaming session...")
    init_start = time.time()
    
    streaming_inference_session = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    
    init_time = time.time() - init_start
    print(f"Session initialization took: {init_time:.3f}s")
    
    # Add text prompt
    print(f"Adding text prompt: '{text_prompt}'")
    prompt_start = time.time()
    streaming_inference_session = processor.add_text_prompt(
        inference_session=streaming_inference_session,
        text=text_prompt,
    )
    prompt_time = time.time() - prompt_start
    print(f"Text prompt processing took: {prompt_time:.3f}s")
    
    # Process frames one by one (streaming mode)
    print(f"\nProcessing {len(frames)} frames in streaming mode...")
    process_start = time.time()
    streaming_outputs_per_frame = {}
    frame_times = []
    
    for frame_idx, frame in enumerate(frames):
        frame_start = time.time()
        
        # Pre-resize frame if requested
        if resize_frames:
            frame = resize_image(frame, max_size=720)
        
        # Process frame using the processor
        inputs = processor(images=frame, device=device, return_tensors="pt")
        
        # Process frame using streaming inference
        model_outputs = model(
            inference_session=streaming_inference_session,
            frame=inputs.pixel_values[0],  # Provide processed frame - enables streaming
            reverse=False,
        )
        
        # Post-process outputs with original_sizes for proper resolution handling
        processed_outputs = processor.postprocess_outputs(
            streaming_inference_session,
            model_outputs,
            original_sizes=inputs.original_sizes,  # Required for streaming
        )
        
        streaming_outputs_per_frame[frame_idx] = processed_outputs
        
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        
        if (frame_idx + 1) % 10 == 0:
            elapsed = time.time() - process_start
            fps = (frame_idx + 1) / elapsed
            print(f"  Processed {frame_idx + 1}/{len(frames)} frames ({fps:.2f} FPS, last frame: {frame_time*1000:.1f}ms)")
    
    process_time = time.time() - process_start
    total_time = init_time + prompt_time + process_time
    avg_fps = len(frames) / process_time
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"Total frames processed: {len(streaming_outputs_per_frame)}")
    print(f"Session init time: {init_time:.3f}s")
    print(f"Text prompt time: {prompt_time:.3f}s")
    print(f"Frame processing time: {process_time:.3f}s")
    print(f"  Mean per-frame time: {statistics.mean(frame_times)*1000:.1f}ms")
    print(f"  Median per-frame time: {statistics.median(frame_times)*1000:.1f}ms")
    print(f"  Min per-frame time: {min(frame_times)*1000:.1f}ms")
    print(f"  Max per-frame time: {max(frame_times)*1000:.1f}ms")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    
    # Show results for first frame
    if 0 in streaming_outputs_per_frame:
        frame_0 = streaming_outputs_per_frame[0]
        print(f"\nFirst frame results:")
        print(f"  Objects detected: {len(frame_0['object_ids'])}")
        print(f"  Object IDs: {frame_0['object_ids'].tolist()}")
        print(f"  Scores: {[f'{s:.3f}' for s in frame_0['scores'].tolist()]}")
        print(f"  Boxes shape (XYXY absolute coords): {frame_0['boxes'].shape}")
        print(f"  Masks shape: {frame_0['masks'].shape}")
    
    return {
        'outputs': streaming_outputs_per_frame,
        'init_time': init_time,
        'prompt_time': prompt_time,
        'process_time': process_time,
        'total_time': total_time,
        'fps': avg_fps,
        'frame_times': frame_times,
        'num_frames': len(streaming_outputs_per_frame)
    }

def main():
    print("="*70)
    print("SAM3 Video Inference - Transformers Library Test")
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
    
    # Test parameters
    text_prompt = "person"
    max_frames = 50
    
    # Test 1: Pre-loaded video inference (original size)
    results_preloaded = test_preloaded_video_inference(
        video_frames, processor, model, 
        text_prompt=text_prompt, 
        max_frames=max_frames,
        resize_frames=False
    )
    
    # Test 2: Pre-loaded video inference (with pre-resize)
    results_preloaded_resized = test_preloaded_video_inference(
        video_frames, processor, model, 
        text_prompt=text_prompt, 
        max_frames=max_frames,
        resize_frames=True
    )
    
    # Test 3: Streaming video inference (original size)
    results_streaming = test_streaming_video_inference(
        video_frames, processor, model, 
        text_prompt=text_prompt, 
        max_frames=max_frames,
        resize_frames=False
    )
    
    # Test 4: Streaming video inference (with pre-resize)
    results_streaming_resized = test_streaming_video_inference(
        video_frames, processor, model, 
        text_prompt=text_prompt, 
        max_frames=max_frames,
        resize_frames=True
    )
    
    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print("\nPre-loaded Video Inference:")
    print(f"  Original size: {results_preloaded['fps']:.2f} FPS (process: {results_preloaded['process_time']:.2f}s, total: {results_preloaded['total_time']:.2f}s)")
    print(f"  With resize:   {results_preloaded_resized['fps']:.2f} FPS (process: {results_preloaded_resized['process_time']:.2f}s, total: {results_preloaded_resized['total_time']:.2f}s)")
    print(f"  Speedup: {results_preloaded_resized['fps'] / results_preloaded['fps']:.2f}x")
    
    print("\nStreaming Video Inference:")
    print(f"  Original size: {results_streaming['fps']:.2f} FPS (process: {results_streaming['process_time']:.2f}s, total: {results_streaming['total_time']:.2f}s)")
    print(f"  With resize:   {results_streaming_resized['fps']:.2f} FPS (process: {results_streaming_resized['process_time']:.2f}s, total: {results_streaming_resized['total_time']:.2f}s)")
    print(f"  Speedup: {results_streaming_resized['fps'] / results_streaming['fps']:.2f}x")
    
    print("\nMode Comparison (with resize):")
    print(f"  Pre-loaded: {results_preloaded_resized['fps']:.2f} FPS")
    print(f"  Streaming:  {results_streaming_resized['fps']:.2f} FPS")
    print(f"  Ratio: {results_preloaded_resized['fps'] / results_streaming_resized['fps']:.2f}x")
    
    print(f"\n{'='*70}")
    print("✓ All tests completed successfully!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
