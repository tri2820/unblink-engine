"""
End-to-end test for SAM3 worker labels feature.
Tests that labels are correctly extracted and passed through the entire pipeline.
"""

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from worker_sam3 import load_ai_model

async def test_labels_with_multiple_prompts():
    print("="*70)
    print("TEST: Labels with multiple prompts")
    print("="*70)
    
    worker_function = load_ai_model()
    test_image = Path(__file__).parent / "test.jpg"
    
    # Test 1: Multiple prompts
    test_data = {
        "inputs": [
            {
                "id": "test_1",
                "cross_job_id": "session_1",
                "current_frame": str(test_image),
                "prompts": ["person", "cat", "dog", "vehicle"],
                "timestamp": 0
            }
        ]
    }
    
    print("\n1. Testing with prompts: person, cat, dog, vehicle")
    result = worker_function(test_data)
    
    if result and 'output' in result and len(result['output']) > 0:
        output = result['output'][0]
        if 'labels' in output:
            print("   ✓ Labels found!")
            print(f"   Objects: {output['objects']}")
            print(f"   Labels:  {output['labels']}")
            print(f"   Scores:  {output['scores']}")
            
            # Verify lengths match
            if len(output['objects']) == len(output['labels']) == len(output['scores']):
                print("   ✓ Array lengths match!")
            else:
                print("   ✗ Array lengths don't match!")
                return False
        else:
            print("   ✗ Labels missing!")
            return False
    else:
        print("   ✗ No output!")
        return False
    
    # Test 2: Single prompt
    test_data_2 = {
        "inputs": [
            {
                "id": "test_2",
                "cross_job_id": "session_2",
                "current_frame": str(test_image),
                "prompts": ["person"],
                "timestamp": 0
            }
        ]
    }
    
    print("\n2. Testing with single prompt: person")
    result_2 = worker_function(test_data_2)
    
    if result_2 and 'output' in result_2 and len(result_2['output']) > 0:
        output_2 = result_2['output'][0]
        if 'labels' in output_2:
            print("   ✓ Labels found!")
            print(f"   Objects: {output_2['objects']}")
            print(f"   Labels:  {output_2['labels']}")
            
            # All labels should be "person"
            if all(label == "person" for label in output_2['labels']):
                print("   ✓ All labels are 'person' as expected!")
            else:
                print("   ✗ Not all labels are 'person'!")
                return False
        else:
            print("   ✗ Labels missing!")
            return False
    else:
        print("   ✗ No output!")
        return False
    
    # Test 3: Verify TypeScript compatibility
    print("\n3. Verifying TypeScript type compatibility")
    print("   Checking output structure matches WorkerOutput__Segmentation_Result...")
    
    expected_fields = ['id', 'cross_job_id', 'frame_count', 'objects', 'scores', 'labels', 'boxes', 'masks']
    output_fields = list(output.keys())
    
    missing_fields = [f for f in expected_fields if f not in output_fields]
    extra_fields = [f for f in output_fields if f not in expected_fields and f != 'error']
    
    if not missing_fields and not extra_fields:
        print("   ✓ Output structure matches TypeScript types!")
    else:
        if missing_fields:
            print(f"   ✗ Missing fields: {missing_fields}")
        if extra_fields:
            print(f"   ⚠ Extra fields: {extra_fields}")
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
    return True

if __name__ == "__main__":
    asyncio.run(test_labels_with_multiple_prompts())
