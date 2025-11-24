import asyncio
import time
import json
import torch
import os
from PIL import Image
from ws_client_handler import client_handler

import cv2
import numpy as np
from collections import OrderedDict


# -----------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------
MAX_MEDIA_BUFFERS = 100  # Max number of active streams to track



# -----------------------------------------------------------
# WORKER
# -----------------------------------------------------------
def load_ai_model():
    print("Motion-energy worker loaded (soft-mask with per-pixel sigmoid).")

    # --- TUNABLE PARAMETERS FOR PIXEL-LEVEL SENSITIVITY ---

    # 1. PIXEL SENSITIVITY MIDPOINT (0-255)
    # What level of pixel brightness change should be the center of our
    # sensitivity curve (resulting in a 0.5 contribution)?
    # This acts as a soft, adaptive threshold.
    # Good starting value: 30.0
    pixel_midpoint = 30.0

    # 2. PIXEL SENSITIVITY STEEPNESS
    # How sharply do we distinguish between noise and motion?
    # A higher value makes the transition very sharp (more like a threshold).
    # A lower value makes it more gradual and smooth.
    # Note: This value is much smaller than the previous steepness.
    # Good starting value: 0.2
    pixel_steepness = 0.2
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    
    # Map: media_id -> last_frame_gray (numpy array)
    media_buffers = OrderedDict()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def worker_function(data):
        # print("[AI Thread] Computing motion energy...", data)
        inputs_list = data.get("inputs", [])
        outputs = []

        for inp in inputs_list:
            input_id = inp.get("id", "unknown")
            media_id = inp.get("media_id")
            curr_path = inp.get("current_frame")

            if not media_id or not curr_path:
                outputs.append({"id": input_id, "error": "Missing media_id or current_frame"})
                continue

            try:
                # Load current frame
                img = cv2.imread(curr_path)
                if img is None:
                    # If we can't read the current frame, we can't do anything
                    outputs.append({"id": input_id, "error": "Failed to read current frame"})
                    continue
                
                curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Check if we have a previous frame for this media_id
                if media_id in media_buffers:
                    prev_gray = media_buffers[media_id]
                    media_buffers.move_to_end(media_id) # Mark as recently used

                    if prev_gray.shape != curr_gray.shape:
                        h = min(prev_gray.shape[0], curr_gray.shape[0])
                        w = min(prev_gray.shape[1], curr_gray.shape[1])
                        prev_gray = cv2.resize(prev_gray, (w, h))
                        curr_gray = cv2.resize(curr_gray, (w, h))

                    # --- "SOFT MASK" SIGMOID LOGIC ---

                    # 1. Compute the absolute difference (0-255) and convert to float
                    diff = cv2.absdiff(prev_gray, curr_gray).astype(np.float32)

                    # 2. Apply the sigmoid function element-wise
                    x = pixel_steepness * (diff - pixel_midpoint)
                    soft_mask = sigmoid(x)

                    # 3. The final score is the mean of this soft mask.
                    energy_score = np.mean(soft_mask)
                else:
                    # First frame for this media_id, no motion energy yet
                    energy_score = 0.0

                # Update buffer with current frame for next time
                media_buffers[media_id] = curr_gray
                media_buffers.move_to_end(media_id)
                
                # Trim buffers if too many active streams
                if len(media_buffers) > MAX_MEDIA_BUFFERS:
                    media_buffers.popitem(last=False)

                outputs.append({
                    "id": input_id,
                    "motion_energy": float(energy_score)
                })

            except Exception as e:
                print(f"[AI Thread] Error with {e}", inp)
                outputs.append({"id": input_id, "error": str(e)})

        # print("[AI Thread] Motion energy computation finished.")
        return {"output": outputs}

    return worker_function


if __name__ == "__main__":
    worker_function = load_ai_model()
    asyncio.run(client_handler(worker_function))