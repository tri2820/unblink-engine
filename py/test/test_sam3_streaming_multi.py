"""
Test SAM3 streaming video inference with multiple object detection.

This test focuses on streaming mode where frames arrive one at a time,
and tracks multiple object types simultaneously (e.g., person, cat, dog).

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

def test_streaming_multi_object(video_frames, processor, model, text_prompts, max_frames=50):
    """
    Test streaming video inference with multiple object types.
    
    Args:
        video_frames: List of video frames
        processor: Sam3VideoProcessor
        model: Sam3VideoModel
        text_prompts: List of text prompts to track (e.g., ["person", "cat", "dog"])
        max_frames: Maximum number of frames to process
    """
    print(f"\n{'='*70}")
    print(f"TEST: Streaming Multi-Object Detection")
    print(f"{'='*70}")
    print(f"Text prompts: {text_prompts}")
    print(f"Max frames: {max_frames}")
    print("⚠️  Note: Streaming inference may have more false positives and duplicates")
    print("    (hotstart heuristics disabled - requires future frames)")
    
    frames = video_frames[:max_frames]
    
    # Initialize streaming session
    print(f"\nInitializing streaming session...")
    init_start = time.time()
    
    stream_session = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    
    init_time = time.time() - init_start
    print(f"Session initialization took: {init_time:.3f}s")
    
    # Add multiple text prompts to the session
    print(f"\nAdding {len(text_prompts)} text prompts...")
    prompt_start = time.time()
    
    for prompt in text_prompts:
        print(f"  Adding prompt: '{prompt}'")
        stream_session = processor.add_text_prompt(
            inference_session=stream_session,
            text=prompt,
        )
    
    prompt_time = time.time() - prompt_start
    print(f"All text prompts added in: {prompt_time:.3f}s")
    
    # Process frames in streaming mode
    print(f"\nProcessing {len(frames)} frames in streaming mode...")
    process_start = time.time()
    streaming_outputs = {}
    frame_times = []
    
    # Track statistics per prompt
    detections_per_prompt = {prompt: [] for prompt in text_prompts}
    
    for frame_idx, frame in enumerate(frames):
        frame_start = time.time()
        
        # Process frame using the processor
        inputs = processor(images=frame, device=device, return_tensors="pt")
        
        # Process frame using streaming inference
        model_outputs = model(
            inference_session=stream_session,
            frame=inputs.pixel_values[0],  # Provide processed frame - enables streaming
            reverse=False,
        )
        
        # Post-process outputs with original_sizes for proper resolution handling
        processed_outputs = processor.postprocess_outputs(
            stream_session,
            model_outputs,
            original_sizes=inputs.original_sizes,  # Required for streaming
        )
        
        streaming_outputs[frame_idx] = processed_outputs
        
        # Count detections for this frame
        num_objects = len(processed_outputs['object_ids'])
        
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        
        if (frame_idx + 1) % 10 == 0:
            elapsed = time.time() - process_start
            fps = (frame_idx + 1) / elapsed
            print(f"  Processed {frame_idx + 1}/{len(frames)} frames ({fps:.2f} FPS, last frame: {frame_time*1000:.1f}ms, {num_objects} objects)")
    
    process_time = time.time() - process_start
    total_time = init_time + prompt_time + process_time
    avg_fps = len(frames) / process_time
    
    # Analyze results
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"Total frames processed: {len(streaming_outputs)}")
    print(f"Session init time: {init_time:.3f}s")
    print(f"Text prompts time: {prompt_time:.3f}s ({prompt_time/len(text_prompts)*1000:.1f}ms per prompt)")
    print(f"Frame processing time: {process_time:.3f}s")
    print(f"  Mean per-frame time: {statistics.mean(frame_times)*1000:.1f}ms")
    print(f"  Median per-frame time: {statistics.median(frame_times)*1000:.1f}ms")
    print(f"  Min per-frame time: {min(frame_times)*1000:.1f}ms")
    print(f"  Max per-frame time: {max(frame_times)*1000:.1f}ms")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    
    # Analyze detections across all frames
    print(f"\n{'='*70}")
    print("DETECTION ANALYSIS:")
    print(f"{'='*70}")
    
    # Count total unique object IDs across all frames
    all_object_ids = set()
    detections_per_frame = []
    
    for frame_idx, outputs in streaming_outputs.items():
        object_ids = outputs['object_ids'].tolist()
        all_object_ids.update(object_ids)
        detections_per_frame.append(len(object_ids))
    
    print(f"\nTotal unique object IDs detected: {len(all_object_ids)}")
    print(f"Object IDs: {sorted(all_object_ids)}")
    print(f"\nDetections per frame:")
    print(f"  Mean: {statistics.mean(detections_per_frame):.1f}")
    print(f"  Median: {statistics.median(detections_per_frame):.1f}")
    print(f"  Min: {min(detections_per_frame)}")
    print(f"  Max: {max(detections_per_frame)}")
    
    # Show detailed results for first few frames
    print(f"\n{'='*70}")
    print("FIRST 3 FRAMES DETAILED RESULTS:")
    print(f"{'='*70}")
    
    for frame_idx in range(min(3, len(streaming_outputs))):
        if frame_idx in streaming_outputs:
            outputs = streaming_outputs[frame_idx]
            print(f"\nFrame {frame_idx}:")
            print(f"  Objects detected: {len(outputs['object_ids'])}")
            print(f"  Object IDs: {outputs['object_ids'].tolist()}")
            print(f"  Scores: {[f'{s:.3f}' for s in outputs['scores'].tolist()]}")
            print(f"  Boxes shape (XYXY absolute coords): {outputs['boxes'].shape}")
            print(f"  Masks shape: {outputs['masks'].shape}")
            
            # Print individual object details
            for i, (obj_id, score, box) in enumerate(zip(
                outputs['object_ids'].tolist(),
                outputs['scores'].tolist(),
                outputs['boxes'].tolist()
            )):
                print(f"    Object {i}: ID={obj_id}, Score={score:.3f}, Box={[int(x) for x in box]}")
    
    # Show detection timeline (every 10th frame)
    print(f"\n{'='*70}")
    print("DETECTION TIMELINE (every 10th frame):")
    print(f"{'='*70}")
    
    for frame_idx in range(0, len(streaming_outputs), 10):
        if frame_idx in streaming_outputs:
            outputs = streaming_outputs[frame_idx]
            obj_ids = outputs['object_ids'].tolist()
            scores = [f'{s:.2f}' for s in outputs['scores'].tolist()]
            print(f"Frame {frame_idx:3d}: {len(obj_ids)} objects - IDs: {obj_ids}, Scores: {scores}")
    
    return {
        'outputs': streaming_outputs,
        'init_time': init_time,
        'prompt_time': prompt_time,
        'process_time': process_time,
        'total_time': total_time,
        'fps': avg_fps,
        'frame_times': frame_times,
        'num_frames': len(streaming_outputs),
        'unique_object_ids': sorted(all_object_ids),
        'detections_per_frame': detections_per_frame,
    }

def main():
    print("="*70)
    print("SAM3 Streaming Multi-Object Detection Test")
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
    
    # Warmup
    print("\n" + "="*70)
    print("WARMUP PHASE")
    print("="*70)
    print("Running warmup iterations to prepare GPU and caches...")
    
    warmup_start = time.time()
    warmup_frames = 5
    warmup_iterations = 3
    
    for iteration in range(warmup_iterations):
        print(f"\nWarmup iteration {iteration + 1}/{warmup_iterations}...")
        
        # Create warmup session
        warmup_session = processor.init_video_session(
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.bfloat16,
        )
        
        # Add a simple prompt
        warmup_session = processor.add_text_prompt(
            inference_session=warmup_session,
            text="person",
        )
        
        # Process a few frames
        for frame_idx in range(warmup_frames):
            frame = video_frames[frame_idx]
            inputs = processor(images=frame, device=device, return_tensors="pt")
            
            model_outputs = model(
                inference_session=warmup_session,
                frame=inputs.pixel_values[0],
                reverse=False,
            )
            
            processed_outputs = processor.postprocess_outputs(
                warmup_session,
                model_outputs,
                original_sizes=inputs.original_sizes,
            )
        
        print(f"  Completed {warmup_frames} frames")
    
    # Synchronize GPU to ensure all warmup operations are complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    warmup_time = time.time() - warmup_start
    print(f"\nWarmup completed in {warmup_time:.2f}s")
    print("GPU and caches are now warmed up for accurate benchmarking")
    
    # Test parameters
    max_frames = 50
    
    # Test 1: Single object (baseline)
    print("\n" + "="*70)
    print("Test 1: Single Object Detection (baseline)")
    print("="*70)
    results_single = test_streaming_multi_object(
        video_frames, processor, model,
        text_prompts=["person"],
        max_frames=max_frames
    )
    
    # Test 2: Two objects
    print("\n" + "="*70)
    print("Test 2: Two Object Types")
    print("="*70)
    results_double = test_streaming_multi_object(
        video_frames, processor, model,
        text_prompts=["person", "cat"],
        max_frames=max_frames
    )
    
    # Test 3: Three objects
    print("\n" + "="*70)
    print("Test 3: Three Object Types")
    print("="*70)
    results_triple = test_streaming_multi_object(
        video_frames, processor, model,
        text_prompts=["person", "cat", "dog"],
        max_frames=max_frames
    )
    
    # Test 4: Many objects
    print("\n" + "="*70)
    print("Test 4: Many Object Types")
    print("="*70)
    results_many = test_streaming_multi_object(
        video_frames, processor, model,
        text_prompts=["person", "cat", "dog", "chair", "bed", "lamp"],
        max_frames=max_frames
    )
    
    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    
    tests = [
        ("1 prompt (person)", results_single),
        ("2 prompts (person, cat)", results_double),
        ("3 prompts (person, cat, dog)", results_triple),
        ("6 prompts (person, cat, dog, chair, bed, lamp)", results_many),
    ]
    
    print("\nPerformance:")
    for name, results in tests:
        print(f"\n{name}:")
        print(f"  FPS: {results['fps']:.2f}")
        print(f"  Process time: {results['process_time']:.2f}s")
        print(f"  Total time: {results['total_time']:.2f}s")
        print(f"  Unique objects detected: {len(results['unique_object_ids'])}")
        print(f"  Object IDs: {results['unique_object_ids']}")
        print(f"  Avg detections per frame: {statistics.mean(results['detections_per_frame']):.1f}")
    
    print(f"\n{'='*70}")
    print("KEY FINDINGS:")
    print(f"{'='*70}")
    
    # Calculate performance impact
    baseline_fps = results_single['fps']
    print(f"\nPerformance impact of multiple prompts:")
    for name, results in tests[1:]:
        fps_ratio = results['fps'] / baseline_fps
        slowdown = (1 - fps_ratio) * 100
        print(f"  {name}: {results['fps']:.2f} FPS ({fps_ratio:.2f}x baseline, {slowdown:+.1f}% {'slower' if slowdown > 0 else 'faster'})")
    
    # Check if multiple objects can be detected
    print(f"\nMulti-object detection capability:")
    for name, results in tests:
        num_unique = len(results['unique_object_ids'])
        max_per_frame = max(results['detections_per_frame'])
        print(f"  {name}: {num_unique} unique objects, max {max_per_frame} per frame")
    
    print(f"\n{'='*70}")
    print("✓ All tests completed successfully!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
