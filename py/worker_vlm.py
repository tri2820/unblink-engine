"""
BATCH INPUT/OUTPUT SHAPE FOR worker_text_generation:

BATCH INPUT FORMAT (Accepts a standard 'messages' array per input ID):
{
  "inputs": [
    {
      "id": "req_001",
      "messages": [
        {
          "role": "system",
          "content": [
            {"type": "text", "text": "You are an AI security assistant. Your task is to identify potential threats in the provided images."}
          ]
        },
        {
          "role": "user",
          "content": [
            {"type": "image", "image": "/path/to/person.jpg"},
            {"type": "image", "image": "/path/to/package.png"},
            {"type": "text", "text": "What is the person doing with the package? Are they in a restricted area?"}
          ]
        }
      ]
    },
    {
      "id": "req_002",
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "image", "image": "/path/to/car.jpg"},
            {"type": "image", "image": "/path/to/entrance.jpg"},
            {"type": "text", "text": "Describe the vehicle. Is the entrance blocked?"}
          ]
        }
      ]
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
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os
import cv2

def load_ai_model():
    # Load model with optimizations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Read model name from environment variable, default to a smaller model
    model_id = os.getenv("MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    print(f"Loading {model_id} model on {device}...")
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Reduce image size while maintaining aspect ratio
    print(f"Original image size: {processor.image_processor.size}")
    # Reduce image size for faster processing
    processor.image_processor.size = {"longest_edge": 600}
    print(f"Optimized image size: {processor.image_processor.size}")

    # Optimization trick: Optimal model configuration with float16
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    def worker_function(data):
        """Simulates a long-running, CPU/GPU-intensive task on the client machine."""
        print(f"[AI Thread] Starting heavy AI workload with data: {data}")

        # Prepare optimized batch of messages
        # --- MODIFICATION START ---
        # The new format passes the 'messages' array directly, which is what the processor expects.
        # This simplifies the logic significantly.
        batch_for_processor = []
        message_inputs = data.get('inputs', [])
        for inp in message_inputs:
            # Directly append the messages list from the input
            if 'messages' in inp and isinstance(inp['messages'], list):
                batch_for_processor.append(inp['messages'])
            else:
                print(f"[AI Thread] Warning: Input with id '{inp.get('id')}' is missing a 'messages' list. Skipping.")
        # --- MODIFICATION END ---

        if not batch_for_processor:
            print("[AI Thread] No valid messages found in the input. Aborting workload.")
            return json.dumps({"output": []})

        # Build inputs (processor returns a dict of tensors)
        inputs = processor.apply_chat_template(
            batch_for_processor,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        inputs = inputs.to('cuda')

        raw_outputs = model.generate(**inputs, max_new_tokens=256)

        outputs = []
        i = 0
        for raw_output in raw_outputs:
            tok_ids = raw_output.cpu().tolist()
            raw_text = processor.decode(tok_ids, skip_special_tokens=True)
            # Keep previous logic for extracting assistant reply
            response = raw_text.split("Assistant: ")[-1].strip()
            # Include image name in output
            outputs.append({
                "id": message_inputs[i]['id'],
                "response": response
            })
            i += 1

        result = { "output": outputs }
        print("[AI Thread] Heavy AI workload finished.")
        return result

    return worker_function

if __name__ == "__main__":
    worker_function = load_ai_model()
    asyncio.run(client_handler(worker_function))