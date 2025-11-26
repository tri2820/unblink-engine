"""
Test script for SigLIP2 embeddings model
https://huggingface.co/google/siglip2-base-patch16-224
"""

import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
from io import BytesIO

def test_siglip_basic():
    """Test basic SigLIP2 functionality with text and images"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP2 model on {device}...")
    
    # Load model and processor
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to(device)
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    
    print(f"Model loaded successfully!")
    print(f"Model config: {model.config}")
    
    # Test with sample text
    texts = ["a photo of a cat", "a photo of a dog", "a beautiful sunset"]
    
    # Test with sample image (download from web)
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    print("\n=== Testing Text Embeddings ===")
    text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_outputs = model.get_text_features(**text_inputs)
        # Normalize embeddings
        text_embeddings = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
    
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Sample text embedding (first 10 dims): {text_embeddings[0][:10].tolist()}")
    
    print("\n=== Testing Image Embeddings ===")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_outputs = model.get_image_features(**image_inputs)
        # Normalize embeddings
        image_embeddings = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
    
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Sample image embedding (first 10 dims): {image_embeddings[0][:10].tolist()}")
    
    print("\n=== Testing Similarity Scores ===")
    # Compute similarity between image and texts
    similarities = (image_embeddings @ text_embeddings.T) * 100  # Scale by 100 for readability
    print(f"Image-text similarities: {similarities[0].tolist()}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_siglip_basic()
