import torch
from transformers import AutoModelForCausalLM
from dotenv import load_dotenv
import os

# Load environment variables from .env file (looks in parent directories automatically)
load_dotenv()

# Set HF token for downloading models
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

moondream = AutoModelForCausalLM.from_pretrained(
    "moondream/moondream3-preview",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map={"": "cuda"},
)
moondream.compile()

from PIL import Image

# Simple VQA
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "test.jpg")
image = Image.open(image_path)
result = moondream.query(image=image, question="What's in this image?", reasoning=False)
print(result["answer"])

# Different caption lengths
image = Image.open(image_path)

# Short caption
short = moondream.caption(image, length="short")
print(f"Short: {short['caption']}")

# Normal caption (default)
normal = moondream.caption(image, length="normal")
print(f"Normal: {normal['caption']}")

# Long caption
long = moondream.caption(image, length="long")
print(f"Long: {long['caption']}")