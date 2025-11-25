import sys
import os
import json

# Add parent directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from worker_vlm import load_ai_model

def test_worker_vlm(worker_func):
    # 1. Setup test image
    image_path = os.path.join(current_dir, "test.jpg")
    
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' does not exist.")
        return

    print(f"Using test image: {image_path}")

    print("\n--- Test Case: VLM Query ---")
    
    query_text = "Identify all people in the image. Return a valid JSON array of objects, where each object has keys: 'type' (string, e.g., 'person') and 'is_suspicious' (boolean). Do not include markdown formatting."
    
    data_query = {
        "inputs": [
            {
                "id": "test_vlm_query",
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an AI security assistant. Your task is to identify potential threats in the provided images."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": query_text}
                        ]
                    }
                ]
            }
        ]
    }
    
    try:
        print(f"Sending query: {query_text}")
        result = worker_func(data_query)
        print("\nResult:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error running worker: {e}")

def test_worker_vlm_summary(worker_func):
    # 1. Setup test image
    image_path = os.path.join(current_dir, "test.jpg")
    
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' does not exist.")
        return

    print(f"Using test image: {image_path}")

    print("\n--- Test Case: VLM Summary ---")
    
    query_text = "Provide a concise title and description for this image. Return in JSON format: {title, description}"
    
    data_query = {
        "inputs": [
            {
                "id": "test_vlm_summary",
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an AI assistant that provides concise titles and descriptions for images."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": query_text}
                        ]
                    }
                ]
            }
        ]
    }
    
    try:
        print(f"Sending query: {query_text}")
        result = worker_func(data_query)
        print("\nResult:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error running worker: {e}")

def test_worker_vlm_multi_image(worker_func):
    # 1. Setup test images
    image_path = os.path.join(current_dir, "test.jpg")
    
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' does not exist.")
        return

    print(f"Using test image: {image_path} (repeated 3 times for multi-image test)")

    print("\n--- Test Case: VLM Multi-Image Title and Description ---")
    
    query_text = "Analyze these three images and provide a single title and description that encompasses all of them. Return in JSON format: {\"title\": \"string\", \"description\": \"string\"}"
    
    data_query = {
        "inputs": [
            {
                "id": "test_vlm_multi_image",
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an AI assistant that analyzes multiple images and provides unified titles and descriptions."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "image", "image": image_path},
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": query_text}
                        ]
                    }
                ]
            }
        ]
    }
    
    try:
        print(f"Sending query: {query_text}")
        result = worker_func(data_query)
        print("\nResult:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error running worker: {e}")

def test_worker_vlm_batch_inputs(worker_func):
    # 1. Setup test images
    image_path = os.path.join(current_dir, "test.jpg")
    
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' does not exist.")
        return

    print(f"Using test image: {image_path} (for batch processing test)")

    print("\n--- Test Case: VLM Batch Inputs ---")
    
    data_query = {
        "inputs": [
            {
                "id": "batch_input_1",
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an AI assistant that analyzes multiple images and provides unified titles and descriptions."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "image", "image": image_path},
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": "Analyze these images and provide a single title and description that encompasses all of them. Your response must be valid JSON containing only the keys 'title' and 'description', with no additional keys, markdown, or extra text. Format: {\"title\": \"string\", \"description\": \"string\"}"}
                        ]
                    }
                ]
            },
            {
                "id": "batch_input_1",
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an AI assistant that analyzes multiple images and provides unified titles and descriptions."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": "Analyze these images and provide a single title and description that encompasses both of them. Your response must be valid JSON containing only the keys 'title' and 'description', with no additional keys, markdown, or extra text. Format: {\"title\": \"string\", \"description\": \"string\"}"}
                        ]
                    }
                ]
            },
            {
                "id": "batch_input_1",
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an AI assistant that analyzes multiple images and provides unified titles and descriptions."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": "Analyze these images and provide a single title and description that encompasses both of them. Your response must be valid JSON containing only the keys 'title' and 'description', with no additional keys, markdown, or extra text. Format: {\"title\": \"string\", \"description\": \"string\"}"}
                        ]
                    }
                ]
            },
            {
                "id": "batch_input_2",
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an AI assistant that analyzes multiple images and provides unified titles and descriptions."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": "Analyze these images and provide a single title and description that encompasses both of them. Your response must be valid JSON containing only the keys 'title' and 'description', with no additional keys, markdown, or extra text. Format: {\"title\": \"string\", \"description\": \"string\"}"}
                        ]
                    }
                ]
            }
        ]
    }
    
    try:
        print("Sending batch query with 3 inputs...")
        result = worker_func(data_query)
        print("\nResult:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error running worker: {e}")

if __name__ == "__main__":
    print("Loading model (this may take a while)...")
    try:
        worker_func = load_ai_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)
    
    # test_worker_vlm(worker_func)
    # test_worker_vlm_summary(worker_func)
    # test_worker_vlm_multi_image(worker_func)
    test_worker_vlm_batch_inputs(worker_func)
