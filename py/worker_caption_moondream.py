"""
BATCH INPUT/OUTPUT SHAPE FOR worker:

BATCH INPUT FORMAT (Accepts a standard 'messages' array per input ID):
{
  "inputs": [
    {
      "id": "req_001",
      "images": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
      "query": "Optional query string"
    }
  ]
}

BATCH OUTPUT FORMAT (A single response is generated in response to the messages for an ID):
{
  "output": [
    {
      "id": "req_001",
      "response": "The model's answer based on the provided system prompt, images, and questions."
    },
    {
      "id": "req_002",
      "response": "The model's answer based on the car/entrance images and the two questions."
    }
  ]
}
"""

import asyncio
from ws_client_handler import client_handler
import time
import json
import torch
import sys
import os

# Use transformers v4.53 for moondream compatibility
script_dir = os.path.dirname(os.path.abspath(__file__))
transformers_v4_path = os.path.join(script_dir, 'transformers-v4.53', 'src')
sys.path.insert(0, transformers_v4_path)

from transformers import AutoModelForCausalLM
import torch
from PIL import Image
import cv2

def load_ai_model():
    # Load model with optimizations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moondream = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream3-preview",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    moondream.compile()

    def worker_function(data):
        """A long-running, CPU/GPU-intensive task on the client machine."""
        print(f"[AI Thread] Starting heavy AI workload with data: {data}")

        outputs = []
        message_inputs = data.get('inputs', [])
        for item in message_inputs:
            # Handle multiple images (strict)
            image_paths = item.get('images', [])
            
            if not image_paths or not isinstance(image_paths, list):
                 outputs.append({
                    "id": item.get('id'),
                    "response": "Error: 'images' list is required."
                })
                 continue

            loaded_images = []
            for p in image_paths:
                if os.path.exists(p):
                    loaded_images.append(Image.open(p))
            
            if not loaded_images:
                outputs.append({
                    "id": item.get('id'),
                    "response": "Error: No valid images found."
                })
                continue

            # Determine resize limit
            # If only 1 image, 500px limit. If multiple, 300px limit per image.
            max_edge = 500 if len(loaded_images) == 1 else 300
            
            resized_images = []
            for img in loaded_images:
                if max(img.size) > max_edge:
                    ratio = max_edge / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    # print(f"[AI Thread] Resized image to {new_size}")
                resized_images.append(img)
            
            # Stitch if multiple
            if len(resized_images) > 1:
                total_width = sum(img.width for img in resized_images)
                max_height = max(img.height for img in resized_images)
                
                stitched = Image.new('RGB', (total_width, max_height))
                x_offset = 0
                for img in resized_images:
                    stitched.paste(img, (x_offset, 0))
                    x_offset += img.width
                image = stitched
                print(f"[AI Thread] Stitched {len(resized_images)} images into {image.size}")
            else:
                image = resized_images[0]

            # Check for query
            query = item.get('query')
            if query:
                print(f"[AI Thread] Running query: {query}")
                result = moondream.query(image=image, question=query)
                print(f"[AI Thread] Query result:", result)
                outputs.append({
                    "id": item.get('id'),
                    "response": result['answer']
                })
            else:
                # Different caption lengths
                long = moondream.caption(image, length="long")
                print(f"[AI Thread] Generated caption:", long)
                outputs.append({
                    "id": item.get('id'),
                    "response": long['caption']
                })

        result = { "output": outputs }
        print("[AI Thread] Heavy AI workload finished.")
        return result

    return worker_function

if __name__ == "__main__":
    worker_function = load_ai_model()
    asyncio.run(client_handler(worker_function))