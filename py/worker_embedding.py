import asyncio
from ws_client_handler import client_handler
import time
import json
import torch
from transformers import AutoModel, AutoProcessor
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
      "prompt_name": "query"                     # or "passage" (ignored by SigLIP)
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
      "embedding": [0.1, 0.2, 0.3, ...]         # Vector of size 768 (SigLIP2 model)
    },
    {
      "id": "text_2",
      "embedding": [0.4, 0.5, 0.6, ...]         # Vector of size 768 (SigLIP2 model)
    },
    {
      "id": "img_1",
      "embedding": [0.7, 0.8, 0.9, ...]         # Vector of size 768 (SigLIP2 model)
    }
  ]
}
"""

def load_ai_model():
    """Initializes the SigLIP2 embeddings model and returns the worker function."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP2 model (google/siglip2-base-patch16-224) on {device}...")
    
    # Initialize the model and processor
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to(device)
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    
    def worker_function(data):
        """Processes embedding requests using the SigLIP2 embeddings model."""
        print(f"[Embedding Thread] Starting embedding workload with data: {data}")
        
        inputs = data.get('inputs', [])
        
        if not inputs:
            raise ValueError("No inputs provided for embedding")
        
        text_inputs = []
        image_inputs = []
        
        # Separate text and image inputs
        for inp in inputs:
            if 'text' in inp:
                text_inputs.append(inp)
            elif 'filepath' in inp:
                image_inputs.append(inp)
        
        result_embeddings = []
        
        # Process text inputs (SigLIP doesn't distinguish between query/passage)
        if text_inputs:
            texts = [inp['text'] for inp in text_inputs]
            ids = [inp['id'] for inp in text_inputs]
            
            # Process texts with SigLIP processor
            text_processed = processor(text=texts, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                text_features = model.get_text_features(**text_processed)
                # Normalize embeddings
                text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)
            
            for i, result_id in enumerate(ids):
                result_embeddings.append({
                    "id": result_id,
                    "embedding": text_embeddings[i].tolist()
                })

        # Process image inputs
        if image_inputs:
            ids = [inp['id'] for inp in image_inputs]
            
            # Load images from file paths
            images = []
            for inp in image_inputs:
                img = Image.open(inp['filepath']).convert('RGB')
                images.append(img)
            
            # Process images with SigLIP processor
            image_processed = processor(images=images, return_tensors="pt").to(device)
            
            with torch.no_grad():
                image_features = model.get_image_features(**image_processed)
                # Normalize embeddings
                image_embeddings = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for i, result_id in enumerate(ids):
                result_embeddings.append({
                    "id": result_id,
                    "embedding": image_embeddings[i].tolist()
                })
        
        result = {
            "output": result_embeddings
        }
        print("[Embedding Thread] Embedding workload finished.")
        return result
            
       

    return worker_function

if __name__ == "__main__":
    worker_function = load_ai_model()
    # This assumes a 'client_handler' function is defined elsewhere to run the worker
    asyncio.run(client_handler(worker_function))