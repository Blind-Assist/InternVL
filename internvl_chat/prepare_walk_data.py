import os
import json
import numpy as np
import io
from datasets import load_dataset
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
DATASET_REPO = "blind-assist/walk-train"
OUTPUT_DIR = "./data/walk_vlm"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
JSONL_PATH = os.path.join(OUTPUT_DIR, "walk_train.jsonl")
FRAMES_PER_VIDEO = 2 

os.makedirs(IMG_DIR, exist_ok=True)

def process_data():
    print(f"Loading dataset: {DATASET_REPO}...")
    # Loading the dataset from Hugging Face
    dataset = load_dataset(DATASET_REPO, split="train")
    
    formatted_data = []
    
    print("Processing videos...")
    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        try:
            # 1. Get the target text (The 'alter' field)
            # We check if 'alter' exists directly or if we need to parse a json string
            target_text = item.get("alter")
            
            # Fallback: if your dataset has the JSON in a 'text' column, parse it:
            if not target_text and "json" in item: 
                 data_json = json.loads(item["json"])
                 target_text = data_json.get("alter")

            # If we still don't have text, skip or add placeholder (optional)
            if not target_text:
                continue

            # 2. Handle Video Input
            # HF datasets usually provide 'video' as bytes or a file path
            video_obj = item.get("video") 
            video_path = f"temp_{idx}.mp4"

            # If video_obj is bytes/memoryview, write to temp file
            if isinstance(video_obj, (bytes, memoryview)):
                with open(video_path, "wb") as f:
                    f.write(video_obj)
            # If it's a dict (HF standard for Video feature), get the path/bytes
            elif isinstance(video_obj, dict) and "bytes" in video_obj:
                 with open(video_path, "wb") as f:
                    f.write(video_obj["bytes"])
            elif isinstance(video_obj, str):
                video_path = video_obj
            else:
                # If we can't identify the video format, skip
                continue

            # 3. Extract Frames
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # Logic: Get frame at 1/3 and 2/3 of the video
            frame_indices = [int(total_frames * (i+1) / (FRAMES_PER_VIDEO + 1)) for i in range(FRAMES_PER_VIDEO)]
            
            for i, frame_idx in enumerate(frame_indices):
                # Convert to PIL Image
                frame_arr = vr[frame_idx].asnumpy()
                image = Image.fromarray(frame_arr)
                
                # Save Image to disk
                image_name = f"video_{idx}_frame_{i}.jpg"
                image.save(os.path.join(IMG_DIR, image_name))
                
                # 4. Create JSONL Entry
                # We use a navigation-specific prompt to match your 'alter' data
                entry = {
                    "id": f"{idx}_{i}",
                    "image": image_name,
                    "width": image.width,
                    "height": image.height,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nAnalyze this scene for navigation hazards."
                        },
                        {
                            "from": "gpt",
                            "value": target_text
                        }
                    ]
                }
                formatted_data.append(entry)

            # Cleanup temp video
            if os.path.exists(f"temp_{idx}.mp4"):
                os.remove(f"temp_{idx}.mp4")

        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")
            continue

    # 5. Save the JSONL file
    with open(JSONL_PATH, "w") as f:
        for entry in formatted_data:
            json.dump(entry, f)
            f.write('\n')
    
    print(f"Done! Processed {len(formatted_data)} frames.")

if __name__ == "__main__":
    process_data()