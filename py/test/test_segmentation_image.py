#!/usr/bin/env python3
"""
Simple test for the image segmentation worker.
Tests basic functionality without WebSocket connection.
"""

import os
import sys
import torch
from PIL import Image
import requests
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from worker_segmentation_image import load_ai_model


def test_basic_segmentation():
    """Test basic image segmentation with a sample image."""
    print("=" * 60)
    print("TEST: Basic Image Segmentation")
    print("=" * 60)
    
    # Load the worker
    print("\n1. Loading worker...")
    worker_function = load_ai_model()
    print("✓ Worker loaded successfully")
    
    # Download a test image
    print("\n2. Downloading test image...")
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # Cats on couch
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    # Save to temporary file
    test_image_path = "/tmp/test_segmentation_image.jpg"
    image.save(test_image_path)
    print(f"✓ Image saved to {test_image_path}")
    print(f"  Image size: {image.size}")
    
    # Test with single prompt
    print("\n3. Testing with single prompt 'cat'...")
    test_data = {
        "inputs": [
            {
                "id": "test_001",
                "cross_job_id": "test_batch",
                "current_frame": test_image_path,
                "prompts": ["cat"]
            }
        ]
    }
    
    result = worker_function(test_data)
    
    # Check output
    assert "output" in result, "Missing 'output' key in result"
    assert len(result["output"]) == 1, "Expected 1 output"
    
    output = result["output"][0]
    print(f"✓ Segmentation complete")
    print(f"  ID: {output['id']}")
    print(f"  Cross Job ID: {output['cross_job_id']}")
    print(f"  Objects detected: {len(output['objects'])}")
    print(f"  Labels: {output['labels']}")
    print(f"  Scores: {output['scores']}")
    print(f"  Boxes: {output['boxes'][:2] if len(output['boxes']) > 2 else output['boxes']}")  # Show first 2
    print(f"  Masks: {len(output['masks'])} masks")
    
    # Verify structure
    assert "error" not in output, f"Got error: {output.get('error')}"
    assert output['id'] == "test_001"
    assert output['cross_job_id'] == "test_batch"
    assert len(output['objects']) > 0, "No objects detected"
    assert len(output['scores']) == len(output['objects'])
    assert len(output['boxes']) == len(output['objects'])
    assert len(output['masks']) == len(output['objects'])
    assert len(output['labels']) == len(output['objects'])
    assert all(label == "cat" for label in output['labels'])
    
    print("\n✓ All assertions passed!")


def test_multiple_prompts():
    """Test with multiple prompts."""
    print("\n" + "=" * 60)
    print("TEST: Multiple Prompts")
    print("=" * 60)
    
    # Load the worker
    print("\n1. Loading worker...")
    worker_function = load_ai_model()
    print("✓ Worker loaded successfully")
    
    # Download a test image with people
    print("\n2. Downloading test image...")
    image_url = "http://images.cocodataset.org/val2017/000000397133.jpg"  # People with objects
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    # Save to temporary file
    test_image_path = "/tmp/test_segmentation_multiple.jpg"
    image.save(test_image_path)
    print(f"✓ Image saved to {test_image_path}")
    
    # Test with multiple prompts
    print("\n3. Testing with multiple prompts ['person', 'bottle']...")
    test_data = {
        "inputs": [
            {
                "id": "test_002",
                "cross_job_id": "test_batch_2",
                "current_frame": test_image_path,
                "prompts": ["person", "bottle"]
            }
        ]
    }
    
    result = worker_function(test_data)
    
    # Check output
    output = result["output"][0]
    print(f"✓ Segmentation complete")
    print(f"  Objects detected: {len(output['objects'])}")
    print(f"  Labels: {output['labels']}")
    print(f"  Scores: {output['scores']}")
    
    # Verify structure
    assert "error" not in output, f"Got error: {output.get('error')}"
    assert len(output['objects']) > 0, "No objects detected"
    assert all(label in ["person", "bottle"] for label in output['labels'])
    
    print("\n✓ All assertions passed!")


def test_batch_processing():
    """Test processing multiple images in a batch."""
    print("\n" + "=" * 60)
    print("TEST: Batch Processing")
    print("=" * 60)
    
    # Load the worker
    print("\n1. Loading worker...")
    worker_function = load_ai_model()
    print("✓ Worker loaded successfully")
    
    # Download test images
    print("\n2. Downloading test images...")
    images = [
        ("http://images.cocodataset.org/val2017/000000039769.jpg", "cat"),  # Cats
        ("http://images.cocodataset.org/val2017/000000397133.jpg", "person"),  # People
    ]
    
    test_inputs = []
    for idx, (url, prompt) in enumerate(images):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        test_image_path = f"/tmp/test_batch_{idx}.jpg"
        image.save(test_image_path)
        
        test_inputs.append({
            "id": f"batch_test_{idx}",
            "cross_job_id": f"batch_{idx}",
            "current_frame": test_image_path,
            "prompts": [prompt]
        })
        print(f"  ✓ Image {idx} saved: {prompt}")
    
    # Process batch
    print("\n3. Processing batch...")
    test_data = {"inputs": test_inputs}
    result = worker_function(test_data)
    
    # Check outputs
    assert len(result["output"]) == 2, "Expected 2 outputs"
    
    for idx, output in enumerate(result["output"]):
        print(f"\n  Image {idx}:")
        print(f"    ID: {output['id']}")
        print(f"    Objects: {len(output['objects'])}")
        print(f"    Labels: {output['labels']}")
        print(f"    Scores: {output['scores']}")
        
        assert "error" not in output, f"Got error in output {idx}: {output.get('error')}"
        assert output['id'] == f"batch_test_{idx}"
        assert len(output['objects']) > 0, f"No objects detected in image {idx}"
    
    print("\n✓ All assertions passed!")


def test_error_handling():
    """Test error handling with missing image."""
    print("\n" + "=" * 60)
    print("TEST: Error Handling")
    print("=" * 60)
    
    # Load the worker
    print("\n1. Loading worker...")
    worker_function = load_ai_model()
    print("✓ Worker loaded successfully")
    
    # Test with missing image
    print("\n2. Testing with non-existent image...")
    test_data = {
        "inputs": [
            {
                "id": "error_test",
                "cross_job_id": "error_batch",
                "current_frame": "/tmp/nonexistent_image.jpg",
                "prompts": ["cat"]
            }
        ]
    }
    
    result = worker_function(test_data)
    
    # Check error is reported properly
    output = result["output"][0]
    print(f"✓ Error handled gracefully")
    print(f"  ID: {output['id']}")
    print(f"  Error: {output.get('error', 'No error')}")
    
    assert "error" in output, "Expected error to be reported"
    assert output['id'] == "error_test"
    assert output['cross_job_id'] == "error_batch"
    
    print("\n✓ All assertions passed!")


if __name__ == "__main__":
    try:
        # Run all tests
        test_basic_segmentation()
        test_multiple_prompts()
        test_batch_processing()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
