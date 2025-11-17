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
          "content": "You are an AI assistant. Your task is to answer questions."
        },
        {
          "role": "user",
          "content": "What is the capital of France?"
        }
      ]
    },
    {
      "id": "req_002",
      "messages": [
        {
          "role": "user",
          "content": "Explain quantum computing in simple terms."
        }
      ]
    }
  ]
}

BATCH OUTPUT FORMAT (Raw model output for each message sequence):
{
  "output": [
    {
      "id": "req_001",
      "response": "The model's answer to the questions."
    },
    {
      "id": "req_002",
      "response": "The model's answer to the quantum computing question."
    }
  ]
}
"""

import asyncio
from ws_client_handler import client_handler
import time
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

def load_ai_model():
    """
    Initializes the LLM model and returns a worker function that processes messages.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using Phi-3 4k instruct, a powerful and available model.
    model_name = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
    
    print(f"Loading {model_name} model on {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    def worker_function(data):
        """Processes a batch of messages and returns raw model output."""
        print(f"[Text Generation Thread] Starting batch text generation workload...")

        message_inputs = data.get('inputs', [])
        if not message_inputs or not isinstance(message_inputs, list):
            print("[Text Generation Thread] No valid inputs found in the data.")
            return json.dumps({"output": []})

        # Prepare batch of messages for the processor
        batch_for_processor = []
        for inp in message_inputs:
            if 'messages' in inp and isinstance(inp['messages'], list):
                batch_for_processor.append(inp['messages'])
            else:
                print(f"[Text Generation Thread] Warning: Input with id '{inp.get('id')}' is missing a 'messages' list. Skipping.")

        if not batch_for_processor:
            print("[Text Generation Thread] No valid messages found in the input. Aborting workload.")
            return json.dumps({"output": []})

        generation_args = {
            "return_full_text": False,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "max_new_tokens": 20000000000000000,
        }

        print(f"Processing a batch of {len(batch_for_processor)} message sequences...")
        batch_outputs = pipe(batch_for_processor, **generation_args)

        outputs = []
        for i, output in enumerate(batch_outputs):
            raw_text = output[0]['generated_text']
            outputs.append({
                "id": message_inputs[i]['id'],
                "response": raw_text
            })

        result = {"output": outputs}
        print("[Text Generation Thread] Batch text generation workload finished.")
        return result

    return worker_function

if __name__ == "__main__":
    worker_function = load_ai_model()
    asyncio.run(client_handler(worker_function))