#!/usr/bin/env python3
"""
Quick smoke test for the image segmentation worker.
Downloads one image and runs segmentation on it.
"""

import os
import sys
import requests
from PIL import Image
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from worker_segmentation_image import load_ai_model


def main():
    print("Quick Smoke Test - Image Segmentation Worker")
    print("=" * 60)
    
    # Load worker
    print("\nLoading worker...")
    worker_function = load_ai_model()
    
    # Download test image
    print("\nDownloading test image (cats)...")
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    test_image_path = "/tmp/quick_test.jpg"
    image.save(test_image_path)
    print(f"Image saved: {test_image_path} ({image.size[0]}x{image.size[1]})")
    
    # Run segmentation
    print("\nRunning segmentation with prompt 'cat'...")
    result = worker_function({
        "inputs": [{
            "id": "quick_test",
            "cross_job_id": "smoke_test",
            "current_frame": test_image_path,
            "prompts": ["cat"]
        }]
    })
    
    # Display results
    output = result["output"][0]
    if "error" in output:
        print(f"\n❌ ERROR: {output['error']}")
        sys.exit(1)
    
    print(f"\n✅ SUCCESS!")
    print(f"   Found {len(output['objects'])} cats")
    print(f"   Scores: {output['scores']}")
    print(f"   Labels: {output['labels']}")
    print(f"   Boxes (XYXY): {output['boxes']}")
    print(f"   Masks: {len(output['masks'])} RLE-encoded masks")
    
    # Cleanup
    os.remove(test_image_path)
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
