import torch
import time
from transformers.image_utils import load_image
from transformers import DFineForObjectDetection, AutoImageProcessor

def benchmark():
    device = torch.device("cuda")
    print(f"Using device: {device}")

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = load_image(url)

    image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-obj2coco")
    model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-obj2coco").to(device)
    
    batch_sizes = [4, 8, 16, 32, 64, 128]
    results = []

    for bs in batch_sizes:
        print(f"\n--- Testing batch size = {bs} ---")

        images = [image] * bs
        inputs = image_processor(images=images, return_tensors="pt").to(device)

        print("Warmup...")
        with torch.no_grad():
            _ = model(**inputs)

        num_iters = 20
        start = time.time()

        for i in range(num_iters):
            inputs = image_processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            if (i + 1) % 5 == 0:
                print(f"Iteration {i+1}/{num_iters}")

        end = time.time()

        total = end - start
        fps = (bs * num_iters) / total
        avg_iter = total / num_iters
        results.append((bs, fps, avg_iter))

        print(f"Done batch size {bs}: {fps:.2f} FPS, {avg_iter*1000:.2f} ms/iter")

    print("\n===== Final Results =====")
    print("Batch Size | FPS     | Avg Time/Iter (ms)")
    print("----------|---------|-------------------")
    for bs, fps, avg in results:
        print(f"{bs:<10} | {fps:>7.2f} | {avg*1000:>9.2f}")

if __name__ == "__main__":
    benchmark()

# ===== Final Results =====
# Batch Size | FPS     | Avg Time/Iter (ms)
# ----------|---------|-------------------
# 4          |   38.38 |    104.23
# 8          |   48.98 |    163.34
# 16         |   65.19 |    245.43
# 32         |   67.73 |    472.46
# 64         |   68.56 |    933.53
# 128        |   67.42 |   1898.67