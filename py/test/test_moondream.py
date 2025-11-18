import torch
from transformers import AutoModelForCausalLM

moondream = AutoModelForCausalLM.from_pretrained(
    "moondream/moondream3-preview",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map={"": "cuda"},
)
moondream.compile()

from PIL import Image

# Simple VQA
image = Image.open("./test.jpg")
result = moondream.query(image=image, question="What's in this image?")
print(result["answer"])

# Different caption lengths
image = Image.open("./test.jpg")

# Short caption
short = moondream.caption(image, length="short")
print(f"Short: {short['caption']}")

# Normal caption (default)
normal = moondream.caption(image, length="normal")
print(f"Normal: {normal['caption']}")

# Long caption
long = moondream.caption(image, length="long")
print(f"Long: {long['caption']}")