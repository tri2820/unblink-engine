"""
Test the worker_embedding with SigLIP2 model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker_embedding import load_ai_model
from PIL import Image
import tempfile

def test_worker():
    print("Loading worker function...")
    worker_function = load_ai_model()
    
    # Create a temporary test image
    test_img = Image.new('RGB', (224, 224), color='red')
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    test_img.save(temp_file.name)
    
    try:
        # Test data with text and image inputs
        test_data = {
            "inputs": [
                {
                    "id": "text_1",
                    "text": "a photo of a cat"
                },
                {
                    "id": "text_2", 
                    "text": "a photo of a dog"
                },
                {
                    "id": "img_1",
                    "filepath": temp_file.name
                }
            ]
        }
        
        print("\n=== Testing Worker Function ===")
        print(f"Input data: {test_data}")
        
        result = worker_function(test_data)
        
        print("\n=== Results ===")
        print(f"Number of outputs: {len(result['output'])}")
        
        for item in result['output']:
            print(f"\nID: {item['id']}")
            print(f"Embedding shape: {len(item['embedding'])} dimensions")
            print(f"First 10 values: {item['embedding'][:10]}")
        
        # Verify results
        assert len(result['output']) == 3, "Should have 3 outputs"
        assert all(len(item['embedding']) == 768 for item in result['output']), "All embeddings should be 768-dimensional"
        
        print("\n✅ All tests passed!")
        
    finally:
        # Clean up temp file
        os.unlink(temp_file.name)

if __name__ == "__main__":
    test_worker()
