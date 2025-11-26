"""
Test SAM3 worker with labels output.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from worker_sam3 import load_ai_model

async def test_sam3_labels():
    print("Loading SAM3 worker...")
    worker_function = load_ai_model()
    
    # Test image path
    test_image = Path(__file__).parent / "test.jpg"
    
    # Test input with multiple prompts
    test_data = {
        "inputs": [
            {
                "id": "test_frame_1",
                "cross_job_id": "test_session",
                "current_frame": str(test_image),
                "prompts": ["person", "cat", "dog"],
                "timestamp": 0
            }
        ]
    }
    
    print("\nProcessing test frame with prompts: person, cat, dog")
    result = worker_function(test_data)
    
    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    print(json.dumps(result, indent=2))
    
    # Check if labels are present
    if result and 'output' in result and len(result['output']) > 0:
        output = result['output'][0]
        if 'labels' in output:
            print("\n✓ Labels found!")
            print(f"  Objects: {output.get('objects', [])}")
            print(f"  Labels:  {output.get('labels', [])}")
            print(f"  Scores:  {output.get('scores', [])}")
        else:
            print("\n✗ Labels missing!")
    else:
        print("\n✗ No output received!")

if __name__ == "__main__":
    asyncio.run(test_sam3_labels())
