import asyncio
import time
import json
import torch
import os
from transformers import AutoImageProcessor, DFineForObjectDetection
from PIL import Image
from ws_client_handler import client_handler


"""
INPUT/OUTPUT SHAPE EXAMPLES:

INPUT FORMAT:
{
  "inputs": [
    {
      "id": "image_group_1",
      "filepath": "path/to/image1.jpg"
    },
    {
      "id": "image_group_2", 
      "filepath": "path/to/another_image.jpg"
    }
  ]
}

OUTPUT FORMAT:
{
  "output": [
    {
      "id": "image_group_1",
      "detections": [
        {
          "label": "person",
          "score": 0.925,
          "box": {
            "x_min": 100.2,
            "y_min": 50.3,
            "x_max": 200.1,
            "y_max": 300.5
          }
        }
      ]
    },
    {
      "id": "image_group_2",
      "detections": [
        {
          "label": "bicycle", 
          "score": 0.754,
          "box": {
            "x_min": 50.1,
            "y_min": 60.2,
            "x_max": 120.3,
            "y_max": 180.4
          }
        }
      ]
    }
  ]
}
"""


def load_ai_model():
    # Load DFine model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = os.getenv("MODEL_ID", "ustc-community/dfine_x_coco")
    print(f"Loading {model_id} on {device}...")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = DFineForObjectDetection.from_pretrained(model_id).to(device)

    def worker_function(data):
        """Run DFine object detection on given images in batch."""
        print("[AI Thread] Starting DFine object detection...", data)
        inputs_list = data.get("inputs", [])
        
        # Pre-load all images to prepare for batch processing
        all_images = []
        all_image_paths = []
        all_input_ids = []
        
        for inp in inputs_list:
            input_id = inp.get("id", "unknown")
            filepath = inp.get("filepath", "")

            if not filepath:
                print(f"[AI Thread] No filepath for {input_id}, skipping.")
                continue

            try:
                img = Image.open(filepath).convert("RGB")
                all_images.append(img)
                all_image_paths.append(filepath)
                all_input_ids.append(input_id)
            except Exception as e:
                print(f"[AI Thread] Failed to open {filepath}: {e}")
                continue

        # Process all images in a single batch if any were loaded
        outputs = []
        if all_images:
            print(f"[AI Thread] Processing {len(all_images)} images in batch...")
            target_sizes = [img.size[::-1] for img in all_images]
            print(f"[AI Thread] Target sizes: {target_sizes}")
            with torch.no_grad():
                inputs = processor(images=all_images, return_tensors="pt").to(device)
                model_outputs = model(**inputs)
                results = processor.post_process_object_detection(
                    model_outputs,
                    threshold=0.3,
                    target_sizes=target_sizes,
                )
                
            # Organize results back to match input structure
            for i, input_id in enumerate(all_input_ids):
                result = results[i]
                image_path = all_image_paths[i]
                
                detections = []
                for box, label, score in zip(result["boxes"], result["labels"], result["scores"]):
                    detections.append({
                        "label": model.config.id2label[label.item()],
                        "confidence": round(score.item(), 3),
                        "box": {
                            "x_min": round(box[0].item(), 2),
                            "y_min": round(box[1].item(), 2),
                            "x_max": round(box[2].item(), 2),
                            "y_max": round(box[3].item(), 2),
                        }
                    })

                outputs.append({"id": input_id, "detections": detections})

        else:
            # If no valid images were loaded, return empty detections for each input
            for inp in inputs_list:
                input_id = inp.get("id", "unknown")
                outputs.append({"id": input_id, "detections": []})

        print("[AI Thread] DFine object detection finished.")
        return {"output": outputs}

    return worker_function


if __name__ == "__main__":
    worker_function = load_ai_model()
    asyncio.run(client_handler(worker_function))
