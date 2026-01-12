import os
import json
import argparse
from datasets import load_dataset
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
DATASET_REPO = "blind-assist/walk-train"
OUTPUT_DIR = "./data/walk_vlm"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
JSONL_PATH = os.path.join(OUTPUT_DIR, "walk_train.jsonl")
META_PATH = os.path.join(OUTPUT_DIR, "walk_meta.json")
FRAMES_PER_VIDEO = 2 

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare WalkVLM Dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process (e.g., 100)")
    return parser.parse_args()

def process_data(limit=None):
    os.makedirs(IMG_DIR, exist_ok=True)
    
    print(f"Loading dataset: {DATASET_REPO}...")
    dataset = load_dataset(DATASET_REPO, split="train")
    
    # Apply limit if requested
    if limit:
        print(f"⚠️ LIMITING dataset to first {limit} videos.")
        dataset = dataset.select(range(limit))

    formatted_data = []
    print("Processing videos...")
    
    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        try:
            target_text = item.get("alter")
            if not target_text and "json" in item and item["json"]:
                try:
                    data_json = json.loads(item["json"])
                    target_text = data_json.get("alter")
                except: pass

            if not target_text: continue 

            video_obj = item.get("video")
            video_path = f"temp_{idx}.mp4"

            if isinstance(video_obj, dict) and "bytes" in video_obj:
                 with open(video_path, "wb") as f: f.write(video_obj["bytes"])
            elif isinstance(video_obj, str):
                video_path = video_obj
            else: continue

            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = [int(total_frames * (i+1) / (FRAMES_PER_VIDEO + 1)) for i in range(FRAMES_PER_VIDEO)]
            
            for i, frame_idx in enumerate(frame_indices):
                frame_arr = vr[frame_idx].asnumpy()
                image = Image.fromarray(frame_arr)
                image_name = f"video_{idx}_frame_{i}.jpg"
                image.save(os.path.join(IMG_DIR, image_name))
                
                entry = {
                    "id": f"{idx}_{i}",
                    "image": image_name,
                    "width": image.width,
                    "height": image.height,
                    "conversations": [
                        { "from": "human", "value": "<image>\nAnalyze this scene for navigation hazards." },
                        { "from": "gpt", "value": target_text }
                    ]
                }
                formatted_data.append(entry)

            if os.path.exists(video_path): os.remove(video_path)

        except Exception as e:
            print(f"Error on index {idx}: {e}")
            continue

    # Save JSONL
    with open(JSONL_PATH, "w") as f:
        for entry in formatted_data:
            json.dump(entry, f)
            f.write('\n')
    
    total_samples = len(formatted_data)
    print(f"✅ Generated {total_samples} training samples.")

    # AUTOMATICALLY GENERATE METADATA
    meta_content = {
        "walk-vlm-custom": {
            "root": "data/walk_vlm/images",
            "annotation": "data/walk_vlm/walk_train.jsonl",
            "data_augment": False,
            "max_dynamic_patch": 6,
            "repeat_time": 1,
            "length": total_samples  # Automatically set correct length
        }
    }
    with open(META_PATH, "w") as f:
        json.dump(meta_content, f, indent=2)
    print(f"✅ Updated metadata at {META_PATH} with length {total_samples}")

if __name__ == "__main__":
    args = parse_args()
    process_data(limit=args.limit)