import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from dotenv import load_dotenv
import os

# Load environment variables from .env file (looks in parent directories automatically)
load_dotenv()

# Set HF token for downloading models
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load an image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "test.jpg")
image = Image.open(image_path)

# Set the image
inference_state = processor.set_image(image)

# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="a person")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

print(f"Found {len(masks)} masks")
print(f"Boxes: {boxes}")
print(f"Scores: {scores}")
