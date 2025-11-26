import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from worker_sam3 import load_ai_model

def decode_rle_mask(rle_data):
    """Decode RLE mask back to 2D numpy array."""
    size = rle_data['size']
    counts = rle_data['counts']
    
    # Reconstruct flat mask
    mask_flat = []
    current_val = 0  # RLE starts with 0s
    for count in counts:
        mask_flat.extend([current_val] * count)
        current_val = 1 - current_val  # Toggle between 0 and 1
    
    # Reshape to 2D
    mask = np.array(mask_flat, dtype=np.uint8).reshape(size)
    return mask

def visualize_result(image_path, result_data, output_path):
    """Visualize detection results with masks and boxes."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Get detection data
    objects = result_data.get('objects', [])
    scores = result_data.get('scores', [])
    boxes = result_data.get('boxes', [])
    masks_rle = result_data.get('masks', [])
    
    if not objects:
        print("  No objects detected, skipping visualization.")
        return
    
    print(f"  Visualizing {len(objects)} objects...")
    
    # Define colors for each object
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
    ]
    
    # Apply masks with colors
    result_img = img_array.copy()
    for i, (obj_id, score, mask_rle) in enumerate(zip(objects, scores, masks_rle)):
        color = colors[i % len(colors)]
        
        # Decode mask
        mask = decode_rle_mask(mask_rle)
        
        # Resize mask to match image size if needed
        if mask.shape != (img_array.shape[0], img_array.shape[1]):
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
            mask = np.array(mask_pil) > 127
        
        # Apply colored overlay
        mask_bool = mask > 0.5
        for c in range(3):
            result_img[:, :, c] = np.where(
                mask_bool,
                result_img[:, :, c] * 0.6 + color[c] * 0.4,
                result_img[:, :, c]
            )
        
        print(f"    Object {i}: ID={obj_id}, Score={score}, Color=RGB{color}")
    
    # Convert back to PIL for drawing boxes
    result_pil = Image.fromarray(result_img.astype(np.uint8))
    draw = ImageDraw.Draw(result_pil)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = None
    
    # Draw boxes and labels
    for i, (obj_id, score, box) in enumerate(zip(objects, scores, boxes)):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"ID{obj_id}: {score:.2f}"
        if font:
            bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1), label, fill='white', font=font)
        else:
            text_bbox = draw.textbbox((x1, y1 - 20), label)
            draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
            draw.text((x1, y1 - 20), label, fill='white')
    
    # Save result
    result_pil.save(output_path, quality=95)
    print(f"  ✓ Visualization saved to: {output_path}")

def test_worker():
    # 1. Setup test image
    # Use the output.jpg from test_sam3_visualize.py
    image_path = os.path.join(current_dir, "test.jpg")
    
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' does not exist. Please ensure test.jpg is in the py/test directory.")
        return

    print(f"Using test image: {image_path}")

    print("Loading SAM3 model (this may take a while)...")
    try:
        worker_func = load_ai_model()
    except Exception as e:
        print(f"Failed to load model (expected if no GPU/dependencies): {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*70)
    print("Test Case 1: Single Frame - First Request (New Session)")
    print("="*70)
    data_single = {
        "inputs": [
            {
                "id": "frame_001",
                "cross_job_id": "video_stream_1",
                "current_frame": image_path,
                "prompts": ["person", "cat", "dog"]
            }
        ]
    }
    try:
        result = worker_func(data_single)
        print("Result:")
        import json
        print(json.dumps(result, indent=2))
        
        # Verify structure
        if 'output' in result and len(result['output']) > 0:
            output = result['output'][0]
            if 'error' in output:
                print(f"\n[FAIL] Error in output: {output['error']}")
            else:
                print("\n[PASS] Output structure verified:")
                print(f"  - ID: {output.get('id')}")
                print(f"  - Cross Job ID: {output.get('cross_job_id')}")
                print(f"  - Frame Count: {output.get('frame_count')}")
                print(f"  - Objects detected: {len(output.get('objects', []))}")
                print(f"  - Object IDs: {output.get('objects')}")
                print(f"  - Scores: {output.get('scores')}")
                print(f"  - Boxes count: {len(output.get('boxes', []))}")
                print(f"  - Masks count: {len(output.get('masks', []))}")
                
                # Check mask structure
                if output.get('masks'):
                    mask = output['masks'][0]
                    print(f"\n  First mask structure:")
                    print(f"    - Size: {mask.get('size')}")
                    print(f"    - Counts length: {len(mask.get('counts', []))}")
                
                # Visualize results
                print("\n  Creating visualization...")
                viz_path = os.path.join(current_dir, "test_case_1_visualization.jpg")
                visualize_result(image_path, output, viz_path)
                    
        else:
            print("[FAIL] No output in result")
            
    except Exception as e:
        print(f"Error running worker: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Test Case 2: Second Frame - Same Session")
    print("="*70)
    data_second = {
        "inputs": [
            {
                "id": "frame_002",
                "cross_job_id": "video_stream_1",  # Same cross_job_id
                "current_frame": image_path
                # No prompts - should reuse session
            }
        ]
    }
    try:
        result = worker_func(data_second)
        print("Result:")
        print(json.dumps(result, indent=2))
        
        if 'output' in result and len(result['output']) > 0:
            output = result['output'][0]
            if 'error' not in output:
                print("\n[PASS] Session reused successfully:")
                print(f"  - Frame Count: {output.get('frame_count')} (should be 2)")
                print(f"  - Objects tracked: {len(output.get('objects', []))}")
                
                # Visualize results
                print("\n  Creating visualization...")
                viz_path = os.path.join(current_dir, "test_case_2_visualization.jpg")
                visualize_result(image_path, output, viz_path)
            else:
                print(f"\n[FAIL] Error: {output['error']}")
                
    except Exception as e:
        print(f"Error running worker: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Test Case 3: New Session - Different Stream")
    print("="*70)
    data_new_stream = {
        "inputs": [
            {
                "id": "frame_101",
                "cross_job_id": "video_stream_2",  # Different cross_job_id
                "current_frame": image_path,
                "prompts": ["person"]  # Different prompts
            }
        ]
    }
    try:
        result = worker_func(data_new_stream)
        print("Result:")
        print(json.dumps(result, indent=2))
        
        if 'output' in result and len(result['output']) > 0:
            output = result['output'][0]
            if 'error' not in output:
                print("\n[PASS] New session created:")
                print(f"  - Frame Count: {output.get('frame_count')} (should be 1)")
                print(f"  - Objects detected: {len(output.get('objects', []))}")
                
                # Visualize results
                print("\n  Creating visualization...")
                viz_path = os.path.join(current_dir, "test_case_3_visualization.jpg")
                visualize_result(image_path, output, viz_path)
            else:
                print(f"\n[FAIL] Error: {output['error']}")
                
    except Exception as e:
        print(f"Error running worker: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Test Case 4: Batch Processing - Multiple Frames")
    print("="*70)
    data_batch = {
        "inputs": [
            {
                "id": "frame_003",
                "cross_job_id": "video_stream_1",
                "current_frame": image_path
            },
            {
                "id": "frame_004",
                "cross_job_id": "video_stream_1",
                "current_frame": image_path
            },
            {
                "id": "frame_102",
                "cross_job_id": "video_stream_2",
                "current_frame": image_path
            }
        ]
    }
    try:
        result = worker_func(data_batch)
        print("Result:")
        print(json.dumps(result, indent=2))
        
        if 'output' in result:
            print(f"\n[PASS] Batch processing completed:")
            print(f"  - Total outputs: {len(result['output'])}")
            for output in result['output']:
                if 'error' not in output:
                    print(f"  - {output.get('id')}: cross_job_id={output.get('cross_job_id')}, frame_count={output.get('frame_count')}, objects={len(output.get('objects', []))}")
                else:
                    print(f"  - {output.get('id')}: ERROR - {output.get('error')}")
        else:
            print("[FAIL] No output in result")
            
    except Exception as e:
        print(f"Error running worker: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Test Case 5: Reset Session")
    print("="*70)
    data_reset = {
        "inputs": [
            {
                "id": "frame_005",
                "cross_job_id": "video_stream_1",
                "current_frame": image_path,
                "reset_session": True,  # Force reset
                "prompts": ["person", "chair", "bed"]
            }
        ]
    }
    try:
        result = worker_func(data_reset)
        print("Result:")
        print(json.dumps(result, indent=2))
        
        if 'output' in result and len(result['output']) > 0:
            output = result['output'][0]
            if 'error' not in output:
                print("\n[PASS] Session reset successfully:")
                print(f"  - Frame Count: {output.get('frame_count')} (should be 1 after reset)")
                print(f"  - Objects detected: {len(output.get('objects', []))}")
                
                # Visualize results
                print("\n  Creating visualization...")
                viz_path = os.path.join(current_dir, "test_case_5_visualization.jpg")
                visualize_result(image_path, output, viz_path)
            else:
                print(f"\n[FAIL] Error: {output['error']}")
                
    except Exception as e:
        print(f"Error running worker: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Test Case 6: Error Handling - Missing Parameters")
    print("="*70)
    data_error = {
        "inputs": [
            {
                "id": "frame_error",
                # Missing cross_job_id and current_frame
            }
        ]
    }
    try:
        result = worker_func(data_error)
        print("Result:")
        print(json.dumps(result, indent=2))
        
        if 'output' in result and len(result['output']) > 0:
            output = result['output'][0]
            if 'error' in output:
                print(f"\n[PASS] Error handled correctly: {output['error']}")
            else:
                print("[FAIL] Expected error but got success")
        else:
            print("[FAIL] No output in result")
            
    except Exception as e:
        print(f"Error running worker: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)

if __name__ == "__main__":
    test_worker()
