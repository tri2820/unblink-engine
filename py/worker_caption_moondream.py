"""
BATCH INPUT/OUTPUT SHAPE FOR worker:

BATCH INPUT FORMAT (Accepts a standard 'messages' array per input ID):
{
  "inputs": [
    {
      "id": "req_001",
      "image_path": "/path/to/image1.jpg",
    },
    {
      "id": "req_002",
      "image_path": "/path/to/image2.jpg",
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
from transformers import AutoModelForCausalLM
import torch
from PIL import Image
import os
import cv2

def load_ai_model():
    # Load model with optimizations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moondream = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream3-preview",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    moondream.compile()

    def worker_function(data):
        """A long-running, CPU/GPU-intensive task on the client machine."""
        print(f"[AI Thread] Starting heavy AI workload with data: {data}")

        outputs = []
        message_inputs = data.get('inputs', [])
        for item in message_inputs:
            image_path = item.get('image_path')
            if not image_path or not os.path.exists(image_path):
                outputs.append({
                    "id": item.get('id'),
                    "response": "Error: Image path is invalid or does not exist."
                })
                continue

            image = Image.open(image_path)

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