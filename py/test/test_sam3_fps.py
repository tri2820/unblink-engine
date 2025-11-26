import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from dotenv import load_dotenv
import os
import time
import statistics

# Load environment variables from .env file (looks in parent directories automatically)
load_dotenv()

# Set HF token for downloading models
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# OPTIMIZATION 10: Enable cudnn autotuner for repeated inference
torch.backends.cudnn.benchmark = True

# Load the model
print("Loading model with optimizations...")
model = build_sam3_image_model(compile=False)  # OPTIMIZATION 3: Disabled due to compatibility issues

# OPTIMIZATION 2: Note - FP16 conversion disabled due to processor compatibility
# The processor converts images to float32, causing dtype mismatch with FP16 model
# Would need to modify processor's transform pipeline to support FP16

# OPTIMIZATION 6: Note - Cannot change processor resolution from 1008
# The model's RoPE (Rotary Position Encoding) frequencies are precomputed for 1008x1008
# Changing resolution causes shape mismatch in RoPE attention
processor = Sam3Processor(model, resolution=1008)  # Default resolution

# Load images for testing
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "test.jpg")

print("Loading test images...")
original_image = Image.open(image_path)
print(f"Original image size: {original_image.size}")

# Pre-resize image to 720px longest side
def resize_image(image, max_size=720):
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

resized_image = resize_image(original_image, 720)
print(f"Pre-resized image size: {resized_image.size}")

def run_inference_single(image, num_runs=10, warmup_runs=2):
    """Run single image inference multiple times and return timing statistics
    Note: Excludes image loading/resizing time - only measures processor + model inference
    """
    times = []
    
    # Do warmup runs first (not timed)
    for i in range(warmup_runs):
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt="a person")
        masks = output["masks"]
        print(f"  Warmup {i+1}/{warmup_runs}: Found {len(masks)} masks")
    
    # Now do timed runs
    for i in range(num_runs):
        # Synchronize CUDA before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # Set the image (includes processor transform: resize to 1008x1008)
        inference_state = processor.set_image(image)
        
        # Prompt the model with text
        output = processor.set_text_prompt(state=inference_state, prompt="a person")
        
        # Get the masks, bounding boxes, and scores
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        
        # Synchronize CUDA after inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        print(f"  Run {i+1}/{num_runs}: {elapsed:.4f}s ({1/elapsed:.2f} FPS) - Found {len(masks)} masks")
    
    return times

# Test 1: Original large image (7682x5124) -> processor resizes to 1008x1008
print("\n" + "="*60)
print("TEST 1: Original image (7682x5124)")
print("="*60)
original_times = run_inference_single(original_image, num_runs=10, warmup_runs=2)

# Test 2: Pre-resized image (720x480) -> processor resizes to 1008x1008
print("\n" + "="*60)
print("TEST 2: Pre-resized image (720x480)")
print("="*60)
resized_times = run_inference_single(resized_image, num_runs=10, warmup_runs=2)

# Calculate and display statistics
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print(f"\nOriginal Image (7682x5124 -> processor -> 1008x1008):")
print(f"  Mean time: {statistics.mean(original_times):.4f}s")
print(f"  Median time: {statistics.median(original_times):.4f}s")
print(f"  Std dev: {statistics.stdev(original_times):.4f}s")
print(f"  Min time: {min(original_times):.4f}s")
print(f"  Max time: {max(original_times):.4f}s")
print(f"  Mean FPS: {1/statistics.mean(original_times):.2f}")

print(f"\nPre-resized Image (720x480 -> processor -> 1008x1008):")
print(f"  Mean time: {statistics.mean(resized_times):.4f}s")
print(f"  Median time: {statistics.median(resized_times):.4f}s")
print(f"  Std dev: {statistics.stdev(resized_times):.4f}s")
print(f"  Min time: {min(resized_times):.4f}s")
print(f"  Max time: {max(resized_times):.4f}s")
print(f"  Mean FPS: {1/statistics.mean(resized_times):.2f}")

# Calculate speedup
speedup = statistics.mean(original_times) / statistics.mean(resized_times)
print(f"\n{'='*60}")
print(f"Pre-resizing speedup: {speedup:.2f}x")
print(f"Time saved per image: {(statistics.mean(original_times) - statistics.mean(resized_times))*1000:.1f}ms")
if speedup > 1.05:
    print(f"✓ Pre-resizing DOES help! (processor resize is faster on smaller images)")
elif speedup < 0.95:
    print(f"✗ Pre-resizing HURTS performance!")
else:
    print(f"~ Pre-resizing has negligible effect (< 5% difference)")

print("\n" + "="*60)
print("HYPOTHESIS TEST RESULT:")
print("="*60)
print("Question: Does pre-resizing images (before processor transforms)")
print("          improve performance?")
answer = 'YES' if speedup > 1.05 else 'NO' if speedup < 0.95 else 'NEGLIGIBLE'
print(f"Answer: {answer}")
timing = 'MORE' if speedup > 1.05 else 'LESS' if speedup < 0.95 else 'SIMILAR'
print(f"\nThe processor's internal resize from {original_image.size} -> 1008x1008")
print(f"takes {timing} time than resizing from {resized_image.size} -> 1008x1008")
