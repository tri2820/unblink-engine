import asyncio
import time
import json
import torch
import os
from PIL import Image
from ws_client_handler import client_handler
from collections import OrderedDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator


"""
SAM3 Video Object Tracking Worker

INPUT/OUTPUT SHAPE:

INPUT FORMAT:
{
  "inputs": [
    {
      "id": "frame_123",                           # Unique ID for this specific request
      "cross_job_id": "video_stream_A",            # Identifier for the video stream (persistent session)
      "current_frame": "path/to/frame_123.jpg",    # Path to current frame
      "prompts": ["person", "cat"],                # (Optional) Text prompts for detection
      "reset_session": false                       # (Optional) Force reset session for this cross_job_id
    },
    {
      "id": "frame_124",
      "cross_job_id": "video_stream_A", 
      "current_frame": "path/to/frame_124.jpg"
    }
  ]
}

OUTPUT FORMAT:
{
  "output": [
    {
      "id": "frame_123",
      "cross_job_id": "video_stream_A",
      "frame_count": 1,                           # Frame number in this session
      "objects": [0, 1],                          # Object IDs detected
      "scores": [0.969, 0.977],                   # Confidence scores per object
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
      "id": "frame_124",
      "cross_job_id": "video_stream_A",
      "frame_count": 2,
      "objects": [0, 1],
      "scores": [0.970, 0.978],
      "boxes": [...],
      "masks": [...]
    }
  ]
}

ERROR FORMAT (when something goes wrong):
{
  "output": [
    {
      "id": "frame_123",
      "cross_job_id": "video_stream_A",
      "error": "Failed to read current frame"
    }
  ]
}
"""


# -----------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------
MAX_SESSION_BUFFERS = 100  # Max number of active video streams to track


# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def encode_mask_rle(mask):
    """
    Encode a binary mask using Run-Length Encoding (RLE).
    Returns a dict with 'size' and 'counts'.
    """
    # Convert tensor to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Ensure mask is 2D
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    # Flatten mask to 1D and convert to binary
    mask_flat = (mask.flatten() > 0.5).astype(int)
    
    # Run-length encoding
    changes = []
    current_val = mask_flat[0]
    count = 1
    
    for i in range(1, len(mask_flat)):
        if mask_flat[i] == current_val:
            count += 1
        else:
            changes.append(count)
            current_val = mask_flat[i]
            count = 1
    changes.append(count)
    
    # If mask starts with 0, prepend 0 to counts
    if mask_flat[0] == 0:
        changes = [0] + changes
    
    return {
        "size": list(mask.shape),
        "counts": changes
    }


# -----------------------------------------------------------
# WORKER
# -----------------------------------------------------------
def load_ai_model():
    print("Loading SAM3 video tracking worker...")
    
    # Setup device
    device = Accelerator().device
    print(f"Using device: {device}")
    
    # Enable cudnn autotuner for repeated inference
    torch.backends.cudnn.benchmark = True
    
    # Load model and processor
    print("Loading SAM3 model and processor...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    print("SAM3 model loaded successfully.")
    
    # Warmup
    print("Running warmup...")
    dummy_image = Image.new('RGB', (640, 480))
    warmup_session = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    warmup_session = processor.add_text_prompt(
        inference_session=warmup_session,
        text="person",
    )
    for _ in range(3):
        inputs = processor(images=dummy_image, device=device, return_tensors="pt")
        model_outputs = model(
            inference_session=warmup_session,
            frame=inputs.pixel_values[0],
            reverse=False,
        )
        processor.postprocess_outputs(
            warmup_session,
            model_outputs,
            original_sizes=inputs.original_sizes,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warmup completed.")
    
    # Map: cross_job_id -> session data
    # session data = {
    #     'session': inference_session,
    #     'prompts': ['person', 'cat'],
    #     'frame_count': 0
    # }
    session_buffers = OrderedDict()
    
    def cleanup_session(cross_job_id):
        """Clean up session to free GPU memory."""
        if cross_job_id in session_buffers:
            # Force garbage collection
            del session_buffers[cross_job_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def worker_function(data):
        """Process video frames with SAM3 tracking."""
        # print("[AI Thread] Processing SAM3 tracking...", data)
        inputs_list = data.get("inputs", [])
        outputs = []
        
        for inp in inputs_list:
            input_id = inp.get("id", "unknown")
            cross_job_id = inp.get("cross_job_id")
            curr_path = inp.get("current_frame")
            prompts = inp.get("prompts", ["person"])  # Default to tracking persons
            reset_session = inp.get("reset_session", False)
            
            if not cross_job_id or not curr_path:
                outputs.append({
                    "id": input_id,
                    "cross_job_id": cross_job_id,
                    "error": "Missing cross_job_id or current_frame"
                })
                continue
            
            try:
                # Reset session if requested
                if reset_session and cross_job_id in session_buffers:
                    print(f"[AI Thread] Resetting session for {cross_job_id}")
                    cleanup_session(cross_job_id)
                
                # Check if session exists
                if cross_job_id not in session_buffers:
                    print(f"[AI Thread] Creating new session for {cross_job_id} with prompts: {prompts}")
                    
                    # Create new streaming session
                    session = processor.init_video_session(
                        inference_device=device,
                        processing_device="cpu",
                        video_storage_device="cpu",
                        dtype=torch.bfloat16,
                    )
                    
                    # Add prompts
                    for prompt in prompts:
                        session = processor.add_text_prompt(
                            inference_session=session,
                            text=prompt,
                        )
                    
                    session_buffers[cross_job_id] = {
                        'session': session,
                        'prompts': prompts,
                        'frame_count': 0
                    }
                
                # Get session data
                session_data = session_buffers[cross_job_id]
                session = session_data['session']
                session_buffers.move_to_end(cross_job_id)  # Mark as recently used
                
                # Load and process frame
                frame = Image.open(curr_path).convert("RGB")
                inputs = processor(images=frame, device=device, return_tensors="pt")
                
                # Run inference
                model_outputs = model(
                    inference_session=session,
                    frame=inputs.pixel_values[0],
                    reverse=False,
                )
                
                # Post-process outputs
                processed_outputs = processor.postprocess_outputs(
                    session,
                    model_outputs,
                    original_sizes=inputs.original_sizes,
                )
                
                # Increment frame count
                session_data['frame_count'] += 1
                
                # Encode masks using RLE
                masks_rle = []
                for mask in processed_outputs['masks']:
                    masks_rle.append(encode_mask_rle(mask))
                
                # Prepare output
                outputs.append({
                    "id": input_id,
                    "cross_job_id": cross_job_id,
                    "frame_count": session_data['frame_count'],
                    "objects": processed_outputs['object_ids'].tolist(),
                    "scores": [round(s, 3) for s in processed_outputs['scores'].tolist()],
                    "boxes": [[round(coord, 2) for coord in box] for box in processed_outputs['boxes'].tolist()],
                    "masks": masks_rle
                })
                
                # Trim buffers if too many active sessions
                if len(session_buffers) > MAX_SESSION_BUFFERS:
                    oldest_id = next(iter(session_buffers))
                    print(f"[AI Thread] Evicting oldest session: {oldest_id}")
                    cleanup_session(oldest_id)
                
            except Exception as e:
                print(f"[AI Thread] Error processing {cross_job_id}: {e}")
                import traceback
                traceback.print_exc()
                outputs.append({
                    "id": input_id,
                    "cross_job_id": cross_job_id,
                    "error": str(e)
                })
        
        # print("[AI Thread] SAM3 tracking finished.")
        return {"output": outputs}
    
    return worker_function


if __name__ == "__main__":
    worker_function = load_ai_model()
    asyncio.run(client_handler(worker_function))
