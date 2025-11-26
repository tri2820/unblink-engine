import asyncio
import time
import json
import torch
import numpy as np
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
MAX_FRAMES_PER_SESSION = 300  # Reset session after this many frames to prevent unbounded growth
MAX_FRAMES_TO_KEEP = 30  # Keep only last N processed frames (raw pixel data) - SAM3 uses ~7 for temporal context


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
    
    def cleanup_old_frames(session):
        """Remove old frames from session to prevent unbounded memory growth."""
        if hasattr(session, 'processed_frames') and session.processed_frames is not None:
            # Keep only the most recent frames in processed_frames (the raw frame storage)
            # This is the main memory consumer - each frame is ~1-2MB
            frame_indices = sorted(session.processed_frames.keys())
            if len(frame_indices) > MAX_FRAMES_TO_KEEP:
                # Remove oldest frames from processed_frames only
                # DO NOT remove from output_dict_per_obj as SAM3 needs conditioning frame references
                frames_to_remove = frame_indices[:-MAX_FRAMES_TO_KEEP]
                for frame_idx in frames_to_remove:
                    del session.processed_frames[frame_idx]
    
    def cleanup_session(cross_job_id):
        """Clean up session to free GPU memory."""
        if cross_job_id in session_buffers:
            session_data = session_buffers[cross_job_id]
            # Clear session object explicitly
            if 'session' in session_data:
                session = session_data['session']
                # Clear processed frames that accumulate in memory
                if hasattr(session, 'processed_frames') and session.processed_frames is not None:
                    session.processed_frames.clear()
                # Clear cache
                if hasattr(session, 'cache') and hasattr(session.cache, 'clear_all'):
                    session.cache.clear_all()
                # Reset state
                if hasattr(session, 'reset_state'):
                    session.reset_state()
                del session_data['session']
            # Remove from buffers
            del session_buffers[cross_job_id]
            # Force garbage collection and clear CUDA cache
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def worker_function(data):
        """Process video frames with SAM3 tracking."""
        # print("[AI Thread] Processing SAM3 tracking...", data)
        inputs_list = data.get("inputs", [])
        
        # Filter: only keep the latest input for each cross_job_id
        max_timestamp_per_cross_job_id = {}
        for inp in inputs_list:
            cross_job_id = inp.get("cross_job_id")
            timestamp = inp.get("timestamp", 0)
            if cross_job_id:
                if cross_job_id not in max_timestamp_per_cross_job_id or timestamp > max_timestamp_per_cross_job_id[cross_job_id]:
                    max_timestamp_per_cross_job_id[cross_job_id] = timestamp
        
        # Keep only inputs with the latest timestamp for their cross_job_id
        filtered_inputs = []
        for inp in inputs_list:
            cross_job_id = inp.get("cross_job_id")
            timestamp = inp.get("timestamp", 0)
            if not cross_job_id:
                filtered_inputs.append(inp)  # Keep inputs without cross_job_id
            elif timestamp == max_timestamp_per_cross_job_id.get(cross_job_id):
                filtered_inputs.append(inp)  # Keep only the latest
        
        dropped_count = len(inputs_list) - len(filtered_inputs)
        if dropped_count > 0:
            print(f"[AI Thread] Dropped {dropped_count} stale inputs, processing {len(filtered_inputs)} latest")
        
        inputs_list = filtered_inputs
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
                # Reset session if requested or if it has processed too many frames
                if cross_job_id in session_buffers:
                    session_data = session_buffers[cross_job_id]
                    if reset_session or session_data['frame_count'] >= MAX_FRAMES_PER_SESSION:
                        if session_data['frame_count'] >= MAX_FRAMES_PER_SESSION:
                            print(f"[AI Thread] Auto-resetting session {cross_job_id} after {session_data['frame_count']} frames")
                        else:
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
                
                # Clean up old frames to prevent unbounded memory growth
                cleanup_old_frames(session)
                
                # Periodically clear vision features cache to prevent accumulation
                if session_data['frame_count'] % 100 == 0 and hasattr(session, 'cache'):
                    if hasattr(session.cache, '_vision_features') and len(session.cache._vision_features) > 10:
                        # Keep only the most recent features
                        feature_keys = sorted(session.cache._vision_features.keys())
                        for old_key in feature_keys[:-5]:
                            session.cache._vision_features.pop(old_key, None)
                
                # Encode masks using RLE
                masks_rle = []
                for mask in processed_outputs['masks']:
                    masks_rle.append(encode_mask_rle(mask))
                
                # Prepare output (convert tensors to lists immediately)
                object_ids = processed_outputs['object_ids'].tolist()
                scores = [round(s, 3) for s in processed_outputs['scores'].tolist()]
                boxes = [[round(coord, 2) for coord in box] for box in processed_outputs['boxes'].tolist()]
                
                # Extract labels for each object
                # session.prompts maps prompt_id -> prompt_text (e.g., {0: 'person', 1: 'cat'})
                # processed_outputs['prompt_to_obj_ids'] maps prompt_text -> [object_ids] (e.g., {'person': [0, 1]})
                # We need to create object_id -> label mapping
                labels = []
                if hasattr(session, 'prompts') and 'prompt_to_obj_ids' in processed_outputs:
                    # Create reverse mapping: object_id -> prompt_text
                    obj_to_label = {}
                    for prompt_text, obj_ids_list in processed_outputs['prompt_to_obj_ids'].items():
                        for obj_id in obj_ids_list:
                            obj_to_label[obj_id] = prompt_text
                    
                    # Create labels array matching object_ids order
                    labels = [obj_to_label.get(obj_id, "unknown") for obj_id in object_ids]
                else:
                    # Fallback: use "object" as default label
                    labels = ["object"] * len(object_ids)
                
                outputs.append({
                    "id": input_id,
                    "cross_job_id": cross_job_id,
                    "frame_count": session_data['frame_count'],
                    "objects": object_ids,
                    "scores": scores,
                    "boxes": boxes,
                    "masks": masks_rle,
                    "labels": labels
                })
                
                # Clean up tensors and intermediate data to free memory
                del processed_outputs, model_outputs, inputs, frame, masks_rle
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
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
        
        # Periodic cleanup to prevent memory accumulation
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # print("[AI Thread] SAM3 tracking finished.")
        return {"output": outputs}
    
    return worker_function


if __name__ == "__main__":
    worker_function = load_ai_model()
    asyncio.run(client_handler(worker_function))
