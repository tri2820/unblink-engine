import asyncio
from ws_client_handler import client_handler
import time
import json
import torch
from transformers import AutoModel
from PIL import Image
import requests
from io import BytesIO
import os

"""
INPUT/OUTPUT SHAPE EXAMPLES:

INPUT FORMAT:
{
  "inputs": [
    {
      "id": "text_1",
      "text": "Example text for embedding",      # for text inputs
      "prompt_name": "query"                     # or "passage"
    },
    {
      "id": "text_2",
      "text": "Another example text",
      "prompt_name": "passage"
    },
    {
      "id": "img_1",
      "filepath": "path/to/image.jpg",           # for image inputs
    }
  ]
}

OUTPUT FORMAT:
{
  "output": [                                   # Simple array of embeddings
    {
      "id": "text_1",
      "embedding": [0.1, 0.2, 0.3, ...]         # Vector of size 8192 (Jina model)
    },
    {
      "id": "text_2",
      "embedding": [0.4, 0.5, 0.6, ...]         # Vector of size 8192 (Jina model)
    },
    {
      "id": "img_1",
      "embedding": [0.7, 0.8, 0.9, ...]         # Vector of size 8192 (Jina model)
    }
  ]
}
"""

def load_ai_model():
    """Initializes the Jina embeddings model and returns the worker function."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading jina-embeddings-v4 model on {device}...")
    
    # Initialize the model and move it to the appropriate device
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v4", 
        trust_remote_code=True, 
        torch_dtype=torch.float16
    ).to(device)
    
    def worker_function(data):
        """Processes embedding requests using the Jina embeddings model."""
        print(f"[Embedding Thread] Starting embedding workload with data: {data}")
        
        
        inputs = data.get('inputs', [])
        
        if not inputs:
            raise ValueError("No inputs provided for embedding")
        
        text_inputs_query = []
        text_inputs_passage = []
        image_inputs = []
        
        for inp in inputs:
            if 'text' in inp:
                prompt_name = inp.get('prompt_name', 'passage')
                if prompt_name == 'query':
                    text_inputs_query.append(inp)
                else:
                    text_inputs_passage.append(inp)
            elif 'filepath' in inp:
                image_inputs.append(inp)
        
        result_embeddings = []
        
        # --- Process each category separately for clarity ---

        # 1. Process Text Queries
        if text_inputs_query:
            texts = [inp['text'] for inp in text_inputs_query]
            ids = [inp['id'] for inp in text_inputs_query]
            
            embeddings = model.encode_text(
                texts=texts,
                task="retrieval",
                prompt_name="query"
            )
            
            for i, result_id in enumerate(ids):
                result_embeddings.append({
                    "id": result_id,
                    # FIX: Convert the individual Tensor to a list
                    "embedding": embeddings[i].tolist()
                })

        # 2. Process Text Passages
        if text_inputs_passage:
            texts = [inp['text'] for inp in text_inputs_passage]
            ids = [inp['id'] for inp in text_inputs_passage]
            
            embeddings = model.encode_text(
                texts=texts,
                task="retrieval",
                prompt_name="passage"
            )

            for i, result_id in enumerate(ids):
                result_embeddings.append({
                    "id": result_id,
                    # FIX: Convert the individual Tensor to a list
                    "embedding": embeddings[i].tolist()
                })

        # 3. Process Images
        if image_inputs:
            image_paths = [inp['filepath'] for inp in image_inputs]
            ids = [inp['id'] for inp in image_inputs]

            embeddings = model.encode_image(
                images=image_paths,
                task="retrieval"
            )
            
            for i, result_id in enumerate(ids):
                result_embeddings.append({
                    "id": result_id,
                    # FIX: Convert the individual Tensor to a list
                    "embedding": embeddings[i].tolist()
                })
        result = {
            "output": result_embeddings
        }
        print("[Embedding Thread] Embedding workload finished.")
        return json.dumps(result)
            
       

    return worker_function

if __name__ == "__main__":
    worker_function = load_ai_model()
    # This assumes a 'client_handler' function is defined elsewhere to run the worker
    asyncio.run(client_handler(worker_function))