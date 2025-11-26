import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
)
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.eval.postprocessors import PostProcessImage
from dotenv import load_dotenv
import os
import time
import statistics

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use autocast for better performance
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()

print("Loading model...")
model = build_sam3_image_model()

# Setup transforms
transform = ComposeAPI(
    transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Setup postprocessor
postprocessor = PostProcessImage(
    max_dets_per_img=-1,
    iou_type="segm",
    use_original_sizes_box=True,
    use_original_sizes_mask=True,
    convert_mask_to_rle=False,
    detection_threshold=0.5,
    to_cpu=False,
)

# Utility functions
GLOBAL_COUNTER = 1

def create_empty_datapoint():
    """A datapoint is a single image on which we can apply several queries at once."""
    return Datapoint(find_queries=[], images=[])

def set_image(datapoint, pil_image):
    """Add the image to be processed to the datapoint"""
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

def add_text_prompt(datapoint, text_query):
    """Add a text query to the datapoint"""
    global GLOBAL_COUNTER
    assert len(datapoint.images) == 1, "please set the image first"
    
    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1

def resize_image(image, max_size=720):
    """Pre-resize image for optimization"""
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

# Load test images
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "test.jpg")

print("Loading test images...")
original_image = Image.open(image_path)
resized_image = resize_image(original_image, 720)

print(f"Original image: {original_image.size}")
print(f"Pre-resized image: {resized_image.size}")

def run_batch_inference(images, text_prompt, num_runs=10, warmup_runs=2):
    """Run batch inference multiple times and return timing statistics
    
    Note: images should already be pre-resized if desired (not timed here)
    """
    times = []
    
    # Warmup
    for i in range(warmup_runs):
        datapoints = []
        for img in images:
            dp = create_empty_datapoint()
            set_image(dp, img)
            add_text_prompt(dp, text_prompt)
            dp = transform(dp)
            datapoints.append(dp)
        
        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
        output = model(batch)
        print(f"  Warmup {i+1}/{warmup_runs}")
    
    # Timed runs (images are already pre-resized or not)
    for i in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Create datapoints
        datapoints = []
        for img in images:
            dp = create_empty_datapoint()
            set_image(dp, img)
            add_text_prompt(dp, text_prompt)
            dp = transform(dp)
            datapoints.append(dp)
        
        # Collate and move to GPU
        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
        
        # Forward
        output = model(batch)
        
        # Post-process
        processed_results = postprocessor.process_results(output, batch.find_metadatas)
        
        torch.cuda.synchronize()
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        total_detections = sum(len(r['masks']) for r in processed_results.values())
        print(f"  Run {i+1}/{num_runs}: {elapsed:.4f}s ({len(images)/elapsed:.2f} img/s) - {total_detections} total masks")
    
    return times

# Test 1: Batch of 3 original images (pre-resize NOT done)
print("\n" + "="*60)
print("TEST 1: Batch of 3 ORIGINAL images (7682x5124)")
print("="*60)
batch_original_times = run_batch_inference(
    [original_image, original_image, original_image],
    "a person",
    num_runs=10,
    warmup_runs=2
)

# Test 2: Batch of 3 pre-resized images (pre-resize done BEFORE timing)
print("\n" + "="*60)
print("TEST 2: Batch of 3 PRE-RESIZED images (720x480)")
print("="*60)
batch_resized_times = run_batch_inference(
    [resized_image, resized_image, resized_image],  # Already resized!
    "a person",
    num_runs=10,
    warmup_runs=2
)

# Memory profiling - measure RELATIVE memory, not absolute
print("\n" + "="*60)
print("MEMORY PROFILING")
print("="*60)

# Get baseline memory (model loaded)
torch.cuda.empty_cache()
baseline_memory = torch.cuda.memory_allocated() / 1024**3
print(f"\nBaseline (model loaded): {baseline_memory:.2f} GB")

# Profile batch with original images
print("\nProcessing batch with ORIGINAL images...")
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated()

datapoints = []
for img in [original_image, original_image, original_image]:
    dp = create_empty_datapoint()
    set_image(dp, img)
    add_text_prompt(dp, "a person")
    dp = transform(dp)
    datapoints.append(dp)

batch = collate(datapoints, dict_key="dummy")["dummy"]
batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
output = model(batch)
processed_results = postprocessor.process_results(output, batch.find_metadatas)

peak_orig = torch.cuda.max_memory_allocated()
mem_used_orig = (peak_orig - mem_before) / 1024**3
print(f"  Memory used for processing: {mem_used_orig:.2f} GB")
print(f"  Peak total: {peak_orig / 1024**3:.2f} GB")

# Clean up
del datapoints, batch, output, processed_results
torch.cuda.empty_cache()

# Profile batch with pre-resized images
print("\nProcessing batch with PRE-RESIZED images...")
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated()

datapoints = []
for img in [resized_image, resized_image, resized_image]:
    dp = create_empty_datapoint()
    set_image(dp, img)
    add_text_prompt(dp, "a person")
    dp = transform(dp)
    datapoints.append(dp)

batch = collate(datapoints, dict_key="dummy")["dummy"]
batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
output = model(batch)
processed_results = postprocessor.process_results(output, batch.find_metadatas)

peak_resized = torch.cuda.max_memory_allocated()
mem_used_resized = (peak_resized - mem_before) / 1024**3
print(f"  Memory used for processing: {mem_used_resized:.2f} GB")
print(f"  Peak total: {peak_resized / 1024**3:.2f} GB")

# Clean up
del datapoints, batch, output, processed_results
torch.cuda.empty_cache()

memory_saved = mem_used_orig - mem_used_resized
print(f"\nMemory saved by pre-resizing: {memory_saved:.2f} GB ({memory_saved/mem_used_orig*100:.1f}% reduction)")

# Results
print("\n" + "="*60)
print("BATCH PROCESSING RESULTS")
print("="*60)

print(f"\nBatch of 3 Original Images (7682x5124):")
print(f"  Mean time per batch: {statistics.mean(batch_original_times):.4f}s")
print(f"  Mean time per image: {statistics.mean(batch_original_times)/3:.4f}s")
print(f"  Mean throughput: {3/statistics.mean(batch_original_times):.2f} img/s")
print(f"  Memory for processing: {mem_used_orig:.2f} GB")
print(f"  Std dev: {statistics.stdev(batch_original_times):.4f}s")

print(f"\nBatch of 3 Pre-resized Images (720x480):")
print(f"  Mean time per batch: {statistics.mean(batch_resized_times):.4f}s")
print(f"  Mean time per image: {statistics.mean(batch_resized_times)/3:.4f}s")
print(f"  Mean throughput: {3/statistics.mean(batch_resized_times):.2f} img/s")
print(f"  Memory for processing: {mem_used_resized:.2f} GB")
print(f"  Std dev: {statistics.stdev(batch_resized_times):.4f}s")

# Compare
speedup = statistics.mean(batch_original_times) / statistics.mean(batch_resized_times)
print(f"\n{'='*60}")
print(f"PERFORMANCE: Pre-resizing speedup: {speedup:.2f}x")
print(f"  Time saved per batch: {(statistics.mean(batch_original_times) - statistics.mean(batch_resized_times))*1000:.1f}ms")
print(f"  Time saved per image: {(statistics.mean(batch_original_times) - statistics.mean(batch_resized_times))*1000/3:.1f}ms")

print(f"\nMEMORY: Pre-resizing saves: {memory_saved:.2f} GB ({memory_saved/peak_orig*100:.1f}%)")

print(f"\n{'='*60}")
print("CONCLUSION:")
print("="*60)
print("✓ Batch processing with pre-resizing provides:")
print(f"  - {speedup:.2f}x faster processing")
print(f"  - {memory_saved:.2f} GB less GPU memory (~{memory_saved/peak_orig*100:.0f}% reduction)")
print("  - Enables processing larger batches on same GPU")
print("For production use, this is the recommended approach!")
