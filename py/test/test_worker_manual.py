import sys
import os

# Add parent directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from worker_caption_moondream import load_ai_model

def test_worker():
    # 1. Setup test image
    # Use the existing test.jpg in the current directory
    image_path = os.path.join(current_dir, "test.jpg")
    
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' does not exist. Please ensure test.jpg is in the py/test directory.")
        return

    print(f"Using test image: {image_path}")

    print("Loading model (this may take a while)...")
    try:
        worker_func = load_ai_model()
    except Exception as e:
        print(f"Failed to load model (expected if no GPU/dependencies): {e}")
        return

    print("\n--- Test Case 1: Query Mode ---")
    data_query = {
        "inputs": [
            {
                "id": "test_query",
                "images": [image_path],
                "query": "Focus on abnormal activity, humans, or objects. Ignore generic scenery. Describe specific details of people or vehicles. Do not use phrases like 'The image shows'. Start directly with the subject. Output a valid JSON object with keys: title, short_description, long_description. Example: {\"title\": \"Suspicious person\", \"short_description\": \"A person in a hoodie looking into a car.\", \"long_description\": \"A person wearing a black hoodie and jeans is peering into the window of a parked red sedan.\"}"
            }
        ]
    }
    try:
        result = worker_func(data_query)
        print("Result:", result)
        
        # Verify JSON structure
        response_text = result['output'][0]['response']
        import json
        import re
        try:
            cleaned = re.sub(r'```json|```', '', response_text).strip()
            parsed = json.loads(cleaned)
            print("\n[PASS] JSON Structure Verified:")
            print(json.dumps(parsed, indent=2))
            
            required_keys = ["title", "short_description", "long_description"]
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                print(f"[FAIL] Missing keys: {missing}")
            else:
                print("[PASS] All required keys present.")
                
        except json.JSONDecodeError as e:
            print(f"\n[FAIL] Could not parse JSON: {e}")
            print(f"Raw response: {response_text}")
            
    except Exception as e:
        print(f"Error running worker: {e}")

    print("\n--- Test Case 2: Caption Mode (No Query) ---")
    data_caption = {
        "inputs": [
            {
                "id": "test_caption",
                "images": [image_path]
            }
        ]
    }
    try:
        result = worker_func(data_caption)
        print("Result:", result)
    except Exception as e:
        print(f"Error running worker: {e}")

    print("\n--- Test Case 3: Multiple Images (Stitching) ---")
    # Use the same image twice for stitching test
    data_stitching = {
        "inputs": [
            {
                "id": "test_stitching",
                "images": [image_path, image_path],
                "query": "Focus on abnormal activity, humans, or objects. Ignore generic scenery. Describe specific details of people or vehicles. Do not use phrases like 'The image shows'. Start directly with the subject. Output a valid JSON object with keys: title, short_description, long_description. Example: {\"title\": \"Suspicious person\", \"short_description\": \"A person in a hoodie looking into a car.\", \"long_description\": \"A person wearing a black hoodie and jeans is peering into the window of a parked red sedan.\"}"
            }
        ]
    }
    try:
        result = worker_func(data_stitching)
        print("Result:", result)
    except Exception as e:
        print(f"Error running worker: {e}")

if __name__ == "__main__":
    test_worker()
