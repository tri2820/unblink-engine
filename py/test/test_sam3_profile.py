import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from torchvision.transforms import v2
from dotenv import load_dotenv
import os
import time

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# Load model
print("Loading model...")
model = build_sam3_image_model()
processor = Sam3Processor(model, resolution=1008)

# Load images
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "test.jpg")
original_image = Image.open(image_path)

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

print(f"\nOriginal image: {original_image.size}")
print(f"Resized image: {resized_image.size}")

# Profile the transform pipeline step by step
def profile_transform_pipeline(pil_image, name):
    print(f"\n{'='*60}")
    print(f"Profiling: {name} ({pil_image.size})")
    print(f"{'='*60}")
    
    # Step 1: to_image
    torch.cuda.synchronize()
    t0 = time.time()
    img_tensor = v2.functional.to_image(pil_image)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"1. to_image():        {(t1-t0)*1000:.2f}ms  (shape: {img_tensor.shape})")
    
    # Step 2: to device
    torch.cuda.synchronize()
    t1 = time.time()
    img_tensor = img_tensor.to("cuda")
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"2. to(cuda):          {(t2-t1)*1000:.2f}ms")
    
    # Step 3: ToDtype uint8
    torch.cuda.synchronize()
    t2 = time.time()
    img_tensor = v2.functional.to_dtype(img_tensor, torch.uint8, scale=True)
    torch.cuda.synchronize()
    t3 = time.time()
    print(f"3. to_dtype(uint8):   {(t3-t2)*1000:.2f}ms")
    
    # Step 4: Resize to 1008x1008
    torch.cuda.synchronize()
    t3 = time.time()
    img_tensor = v2.functional.resize(img_tensor, size=(1008, 1008))
    torch.cuda.synchronize()
    t4 = time.time()
    print(f"4. resize(1008x1008): {(t4-t3)*1000:.2f}ms  (shape: {img_tensor.shape})")
    
    # Step 5: ToDtype float32
    torch.cuda.synchronize()
    t4 = time.time()
    img_tensor = v2.functional.to_dtype(img_tensor, torch.float32, scale=True)
    torch.cuda.synchronize()
    t5 = time.time()
    print(f"5. to_dtype(float32): {(t5-t4)*1000:.2f}ms")
    
    # Step 6: Normalize
    torch.cuda.synchronize()
    t5 = time.time()
    img_tensor = v2.functional.normalize(img_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    torch.cuda.synchronize()
    t6 = time.time()
    print(f"6. normalize():       {(t6-t5)*1000:.2f}ms")
    
    total_time = t6 - t0
    print(f"\nTOTAL transform time: {total_time*1000:.2f}ms")
    return total_time

# Profile both
time_original = profile_transform_pipeline(original_image, "Original (7682x5124)")
time_resized = profile_transform_pipeline(resized_image, "Pre-resized (720x480)")

print(f"\n{'='*60}")
print(f"COMPARISON")
print(f"{'='*60}")
print(f"Time difference: {(time_original - time_resized)*1000:.2f}ms")
print(f"Speedup: {time_original/time_resized:.2f}x")

# Calculate data sizes
original_pixels = 7682 * 5124
resized_pixels = 720 * 480
final_pixels = 1008 * 1008

print(f"\n{'='*60}")
print(f"MEMORY ANALYSIS")
print(f"{'='*60}")
print(f"Original image:  {original_pixels:,} pixels ({original_pixels * 3:,} bytes RGB)")
print(f"                 = {original_pixels * 3 / 1024 / 1024:.2f} MB")
print(f"\nPre-resized:     {resized_pixels:,} pixels ({resized_pixels * 3:,} bytes RGB)")
print(f"                 = {resized_pixels * 3 / 1024 / 1024:.2f} MB")
print(f"\nFinal (1008²):   {final_pixels:,} pixels ({final_pixels * 3:,} bytes RGB)")
print(f"                 = {final_pixels * 3 / 1024 / 1024:.2f} MB")

print(f"\n{'='*60}")
print(f"WHAT'S HAPPENING")
print(f"{'='*60}")
print(f"WITHOUT pre-resize:")
print(f"  1. Load 7682×5124 PIL image from disk")
print(f"  2. Convert {original_pixels * 3 / 1024 / 1024:.2f}MB to tensor (CPU)")
print(f"  3. Transfer {original_pixels * 3 / 1024 / 1024:.2f}MB to GPU")
print(f"  4. Resize on GPU to {final_pixels * 3 / 1024 / 1024:.2f}MB")
print(f"\nWITH pre-resize:")
print(f"  1. Load 7682×5124 PIL image from disk")
print(f"  2. Resize in PIL to 720×480 (CPU, fast)")
print(f"  3. Convert {resized_pixels * 3 / 1024 / 1024:.2f}MB to tensor (CPU)")
print(f"  4. Transfer {resized_pixels * 3 / 1024 / 1024:.2f}MB to GPU")
print(f"  5. Resize on GPU to {final_pixels * 3 / 1024 / 1024:.2f}MB")
print(f"\nData reduction: {original_pixels / resized_pixels:.1f}x less data to process!")
print(f"                ({original_pixels * 3 / 1024 / 1024:.2f}MB → {resized_pixels * 3 / 1024 / 1024:.2f}MB)")
