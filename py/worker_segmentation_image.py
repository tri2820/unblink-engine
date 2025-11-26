import asyncio
import torch
import os
from PIL import Image
from ws_client_handler import client_handler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

from transformers import Sam3Processor, Sam3Model
from accelerate import Accelerator


"""
SAM3 Image Segmentation Worker (Simplified)

INPUT/OUTPUT SHAPE:

INPUT FORMAT:
{
  "inputs": [
    {
      "id": "image_123",                           # Unique ID for this specific request
      "cross_job_id": "batch_A",                   # Identifier for the batch (for compatibility)
      "current_frame": "path/to/image_123.jpg",    # Path to image
      "prompts": ["person", "cat"],                # (Optional) Text prompts for detection
    },
    {
      "id": "image_124",
      "cross_job_id": "batch_A", 
      "current_frame": "path/to/image_124.jpg",
      "prompts": ["ear"]
    }
  ]
}

OUTPUT FORMAT:
{
  "output": [
    {
      "id": "image_123",
      "cross_job_id": "batch_A",
      "objects": [0, 1],                          # Object IDs detected
      "scores": [0.969, 0.977],                   # Confidence scores per object
      "labels": ["person", "person"],             # Class labels per object (from prompts)
      "boxes": [                                  # Bounding boxes (XYXY format, absolute coords)
        [145.0, 135.0, 291.0, 404.0],
        [312.0, 0.0, 514.0, 394.0]
      ],
      "masks": [                                  # Segmentation masks (H x W, flattened and RLE-encoded)
        {
          "size": [540, 960],
          "counts": "eNqd..."                     # RLE-encoded mask
        },
        {
          "size": [540, 960],
          "counts": "fOpe..."
        }
      ]
    },
    {
      "id": "image_124",
      "cross_job_id": "batch_A",
      "objects": [0],
      "scores": [0.970],
      "labels": ["ear"],
      "boxes": [...],
      "masks": [...]
    }
  ]
}

ERROR FORMAT (when something goes wrong):
{
  "output": [
    {
      "id": "image_123",
      "cross_job_id": "batch_A",
      "error": "Failed to read image"
    }
  ]
}
"""


# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def encode_mask_rle(mask):
    """
    Encode a binary mask using Run-Length Encoding (RLE).
    Returns a dict with 'size' and 'counts'.
    """
    import numpy as np
    
    # Convert tensor to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Ensure mask is 2D
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    # Flatten mask to 1D and convert to binary
    mask_flat = (mask.flatten() > 0.5).astype(int)
    
    n = len(mask_flat)
    if n == 0:
        return {"size": list(mask.shape), "counts": []}
    
    # Find indices where values change
    changes = np.where(mask_flat[1:] != mask_flat[:-1])[0] + 1
    
    # Add 0 (start) and len (end) to indices
    indices = np.concatenate(([0], changes, [n]))
    
    # Calculate counts
    counts = np.diff(indices)
    
    counts_list = counts.tolist()
    
    # If mask starts with 1, we need to prepend a 0 count for the initial 0s
    # (COCO RLE format expects the first count to be for 0s)
    if mask_flat[0] == 1:
        counts_list = [0] + counts_list
    
    return {
        "size": list(mask.shape),
        "counts": counts_list
    }


# -----------------------------------------------------------
# WORKER
# -----------------------------------------------------------
def load_ai_model():
    print("Loading SAM3 image segmentation worker...")
    
    # Setup device
    device = Accelerator().device
    print(f"Using device: {device}")
    
    # Enable cudnn autotuner for repeated inference
    torch.backends.cudnn.benchmark = True
    
    # Load model and processor
    print("Loading SAM3 model and processor...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("SAM3 model loaded successfully.")
    
    # Warmup
    print("Running warmup...")
    dummy_image = Image.new('RGB', (640, 480))
    for _ in range(3):
        inputs = processor(images=dummy_image, text="person", return_tensors="pt").to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = model(**inputs)
        processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warmup completed.")
    
    def worker_function(data):
        """Process images with SAM3 segmentation."""
        inputs_list = data.get("inputs", [])
        outputs = []
        
        for inp in inputs_list:
            input_id = inp.get("id", "unknown")
            cross_job_id = inp.get("cross_job_id", "default")
            image_path = inp.get("current_frame")
            prompts = inp.get("prompts", ["person"])  # Default to detecting persons
            
            if not image_path:
                outputs.append({
                    "id": input_id,
                    "cross_job_id": cross_job_id,
                    "error": "Missing current_frame"
                })
                continue
            
            try:
                # Load image
                image = Image.open(image_path).convert("RGB")
                
                # Process each prompt separately and collect all results
                all_masks = []
                all_boxes = []
                all_scores = []
                all_labels = []
                object_id_counter = 0
                
                for prompt in prompts:
                    # Prepare inputs
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.bfloat16)
                    
                    # Run inference
                    with torch.no_grad():
                        model_outputs = model(**inputs)
                    
                    # Post-process results
                    results = processor.post_process_instance_segmentation(
                        model_outputs,
                        threshold=0.5,
                        mask_threshold=0.5,
                        target_sizes=inputs.get("original_sizes").tolist()
                    )[0]
                    
                    # Collect results for this prompt
                    num_objects = len(results['masks'])
                    if num_objects > 0:
                        all_masks.extend(results['masks'])
                        all_boxes.extend(results['boxes'])
                        all_scores.extend(results['scores'])
                        all_labels.extend([prompt] * num_objects)
                
                # Encode masks using RLE
                masks_rle = []
                for mask in all_masks:
                    masks_rle.append(encode_mask_rle(mask))
                
                # Prepare output (convert tensors to lists)
                object_ids = list(range(len(all_masks)))
                scores = [round(float(s), 3) for s in all_scores]
                boxes = [[round(float(coord), 2) for coord in box] for box in all_boxes]
                
                outputs.append({
                    "id": input_id,
                    "cross_job_id": cross_job_id,
                    "objects": object_ids,
                    "scores": scores,
                    "boxes": boxes,
                    "masks": masks_rle,
                    "labels": all_labels
                })
                
                # Clean up tensors to free memory
                del inputs, model_outputs, results, all_masks, all_boxes, all_scores, masks_rle, image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"[AI Thread] Error processing {input_id}: {e}")
                import traceback
                traceback.print_exc()
                outputs.append({
                    "id": input_id,
                    "cross_job_id": cross_job_id,
                    "error": str(e)
                })
        
        # Periodic cleanup to prevent memory accumulation
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"output": outputs}
    
    return worker_function


if __name__ == "__main__":
    worker_function = load_ai_model()
    asyncio.run(client_handler(worker_function))
