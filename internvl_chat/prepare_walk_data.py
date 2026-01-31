# # import os
# # import json
# # import argparse
# # from datasets import load_dataset, Video
# # from decord import VideoReader, cpu
# # from PIL import Image
# # from tqdm import tqdm
# # from huggingface_hub import HfFileSystem

# # # --- Configuration ---
# # DATASET_REPO = "blind-assist/walk-train"
# # OUTPUT_DIR = "./data/walk_vlm"
# # IMG_DIR = os.path.join(OUTPUT_DIR, "images")
# # JSONL_PATH = os.path.join(OUTPUT_DIR, "walk_train.jsonl")
# # META_PATH = os.path.join(OUTPUT_DIR, "walk_meta.json")
# # FRAMES_PER_VIDEO = 2 

# # def parse_args():
# #     parser = argparse.ArgumentParser(description="Prepare WalkVLM Dataset")
# #     parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process (e.g., 100)")
# #     return parser.parse_args()

# # def process_data(limit=None):
# #     os.makedirs(IMG_DIR, exist_ok=True)
    
# #     # Initialize Hugging Face FileSystem to handle hf:// paths
# #     fs = HfFileSystem()

# #     print(f"Loading dataset: {DATASET_REPO} (Streaming Mode)...")
# #     dataset = load_dataset(DATASET_REPO, split="train", streaming=True)
    
# #     # Disable decoding to get the path/bytes info
# #     dataset = dataset.cast_column("video", Video(decode=False))

# #     if limit:
# #         print(f"⚠️ LIMITING dataset to first {limit} videos.")
# #         dataset = dataset.take(limit)
# #         total_to_process = limit
# #     else:
# #         total_to_process = None

# #     formatted_data = []
# #     print("Processing videos...")
    
# #     for idx, item in tqdm(enumerate(dataset), total=total_to_process):
# #         video_path = f"temp_{idx}.mp4"
# #         try:
# #             target_text = item.get("alter")
# #             if not target_text and "json" in item and item["json"]:
# #                 try:
# #                     data_json = json.loads(item["json"])
# #                     target_text = data_json.get("alter")
# #                 except: pass

# #             if not target_text: continue 

# #             # --- HANDLE VIDEO DOWNLOAD ---
# #             video_obj = item.get("video")
            
# #             if isinstance(video_obj, dict):
# #                 # Case 1: We have raw bytes (small videos sometimes)
# #                 if video_obj.get("bytes"):
# #                      with open(video_path, "wb") as f: 
# #                          f.write(video_obj["bytes"])
                
# #                 # Case 2: We have a path (hf:// or http:// or local)
# #                 elif video_obj.get("path"):
# #                     v_path = video_obj["path"]
                    
# #                     if v_path.startswith("hf://"):
# #                         # Use HfFileSystem to read the internal HF path
# #                         with fs.open(v_path, "rb") as r:
# #                             with open(video_path, "wb") as w:
# #                                 w.write(r.read())
                                
# #                     elif v_path.startswith("http"):
# #                         # Standard URL download
# #                         import requests
# #                         with open(video_path, 'wb') as f:
# #                             f.write(requests.get(v_path).content)
# #                     else:
# #                         # It's likely a local path (unlikely in streaming, but possible)
# #                         video_path = v_path 
# #             else:
# #                 continue

# #             # --- EXTRACT FRAMES ---
# #             # Now video_path is a real file on the local disk
# #             vr = VideoReader(video_path, ctx=cpu(0))
# #             total_frames = len(vr)
# #             frame_indices = [int(total_frames * (i+1) / (FRAMES_PER_VIDEO + 1)) for i in range(FRAMES_PER_VIDEO)]
            
# #             for i, frame_idx in enumerate(frame_indices):
# #                 frame_arr = vr[frame_idx].asnumpy()
# #                 image = Image.fromarray(frame_arr)
# #                 image_name = f"video_{idx}_frame_{i}.jpg"
# #                 image.save(os.path.join(IMG_DIR, image_name))
                
# #                 entry = {
# #                     "id": f"{idx}_{i}",
# #                     "image": image_name,
# #                     "width": image.width,
# #                     "height": image.height,
# #                     "conversations": [
# #                         { "from": "human", "value": "<image>\nAnalyze this scene for navigation hazards." },
# #                         { "from": "gpt", "value": target_text }
# #                     ]
# #                 }
# #                 formatted_data.append(entry)

# #         except Exception as e:
# #             print(f"Error on index {idx}: {e}")
# #             continue
# #         finally:
# #             # Cleanup temp video file if we created one
# #             if video_path.startswith("temp_") and os.path.exists(video_path):
# #                 os.remove(video_path)

# #     # Save JSONL
# #     with open(JSONL_PATH, "w") as f:
# #         for entry in formatted_data:
# #             json.dump(entry, f)
# #             f.write('\n')
    
# #     total_samples = len(formatted_data)
# #     print(f"✅ Generated {total_samples} training samples.")

# #     # Generate Metadata
# #     meta_content = {
# #         "walk-vlm-custom": {
# #             "root": "data/walk_vlm/images",
# #             "annotation": "data/walk_vlm/walk_train.jsonl",
# #             "data_augment": False,
# #             "max_dynamic_patch": 6,
# #             "repeat_time": 1,
# #             "length": total_samples
# #         }
# #     }
# #     with open(META_PATH, "w") as f:
# #         json.dump(meta_content, f, indent=2)
# #     print(f"✅ Updated metadata at {META_PATH}")

# # if __name__ == "__main__":
# #     args = parse_args()
# #     process_data(limit=args.limit)
















# import os
# import json
# import argparse
# import random
# from datasets import load_dataset, Video
# from decord import VideoReader, cpu
# from PIL import Image
# from tqdm import tqdm
# from huggingface_hub import HfFileSystem

# # --- Configuration ---
# DATASET_REPO = "blind-assist/walk-train"
# OUTPUT_DIR = "./data/walk_vlm"
# IMG_DIR = os.path.join(OUTPUT_DIR, "images")
# TRAIN_JSONL_PATH = os.path.join(OUTPUT_DIR, "walk_train.jsonl")
# VAL_JSONL_PATH = os.path.join(OUTPUT_DIR, "walk_val.jsonl")
# META_PATH = os.path.join(OUTPUT_DIR, "walk_meta.json")
# FRAMES_PER_VIDEO = 2
# VAL_SPLIT_RATIO = 0.1  # 10% for validation

# def parse_args():
#     parser = argparse.ArgumentParser(description="Prepare WalkVLM Dataset")
#     parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process (e.g., 100)")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
#     return parser.parse_args()

# def process_data(limit=None, seed=42):
#     os.makedirs(IMG_DIR, exist_ok=True)
#     random.seed(seed)
    
#     # Initialize Hugging Face FileSystem to handle hf:// paths
#     fs = HfFileSystem()

#     print(f"Loading dataset: {DATASET_REPO} (Streaming Mode)...")
#     dataset = load_dataset(DATASET_REPO, split="train", streaming=True)
    
#     # Disable decoding to get the path/bytes info
#     dataset = dataset.cast_column("video", Video(decode=False))

#     if limit:
#         print(f"⚠️ LIMITING dataset to first {limit} videos.")
#         dataset = dataset.take(limit)
#         total_to_process = limit
#     else:
#         total_to_process = None

#     formatted_data = []
#     print("Processing videos...")
    
#     for idx, item in tqdm(enumerate(dataset), total=total_to_process):
#         video_path = f"temp_{idx}.mp4"
#         try:
#             target_text = item.get("alter")
#             if not target_text and "json" in item and item["json"]:
#                 try:
#                     data_json = json.loads(item["json"])
#                     target_text = data_json.get("alter")
#                 except: pass

#             if not target_text: continue 

#             # --- HANDLE VIDEO DOWNLOAD ---
#             video_obj = item.get("video")
            
#             if isinstance(video_obj, dict):
#                 # Case 1: We have raw bytes (small videos sometimes)
#                 if video_obj.get("bytes"):
#                      with open(video_path, "wb") as f: 
#                          f.write(video_obj["bytes"])
                
#                 # Case 2: We have a path (hf:// or http:// or local)
#                 elif video_obj.get("path"):
#                     v_path = video_obj["path"]
                    
#                     if v_path.startswith("hf://"):
#                         # Use HfFileSystem to read the internal HF path
#                         with fs.open(v_path, "rb") as r:
#                             with open(video_path, "wb") as w:
#                                 w.write(r.read())
                                
#                     elif v_path.startswith("http"):
#                         # Standard URL download
#                         import requests
#                         with open(video_path, 'wb') as f:
#                             f.write(requests.get(v_path).content)
#                     else:
#                         # It's likely a local path (unlikely in streaming, but possible)
#                         video_path = v_path 
#             else:
#                 continue

#             # --- EXTRACT FRAMES ---
#             # Now video_path is a real file on the local disk
#             vr = VideoReader(video_path, ctx=cpu(0))
#             total_frames = len(vr)
#             frame_indices = [int(total_frames * (i+1) / (FRAMES_PER_VIDEO + 1)) for i in range(FRAMES_PER_VIDEO)]
            
#             for i, frame_idx in enumerate(frame_indices):
#                 frame_arr = vr[frame_idx].asnumpy()
#                 image = Image.fromarray(frame_arr)
#                 image_name = f"video_{idx}_frame_{i}.jpg"
#                 image.save(os.path.join(IMG_DIR, image_name))
                
#                 entry = {
#                     "id": f"{idx}_{i}",
#                     "image": image_name,
#                     "width": image.width,
#                     "height": image.height,
#                     "conversations": [
#                         { "from": "human", "value": "<image>\nAnalyze this scene for navigation hazards." },
#                         { "from": "gpt", "value": target_text }
#                     ]
#                 }
#                 formatted_data.append(entry)

#         except Exception as e:
#             print(f"Error on index {idx}: {e}")
#             continue
#         finally:
#             # Cleanup temp video file if we created one
#             if video_path.startswith("temp_") and os.path.exists(video_path):
#                 os.remove(video_path)

#     # --- SPLIT INTO TRAIN/VAL ---
#     random.shuffle(formatted_data)
#     val_size = int(len(formatted_data) * VAL_SPLIT_RATIO)
#     val_data = formatted_data[:val_size]
#     train_data = formatted_data[val_size:]
    
#     print(f"📊 Split: {len(train_data)} train, {len(val_data)} validation samples")

#     # Save Train JSONL
#     with open(TRAIN_JSONL_PATH, "w") as f:
#         for entry in train_data:
#             json.dump(entry, f)
#             f.write('\n')
    
#     # Save Validation JSONL
#     with open(VAL_JSONL_PATH, "w") as f:
#         for entry in val_data:
#             json.dump(entry, f)
#             f.write('\n')
    
#     print(f"✅ Generated {len(train_data)} training samples.")
#     print(f"✅ Generated {len(val_data)} validation samples.")

#     # Generate Metadata with both train and val
#     meta_content = {
#         "walk-vlm-train": {
#             "root": "data/walk_vlm/images",
#             "annotation": "data/walk_vlm/walk_train.jsonl",
#             "data_augment": False,
#             "max_dynamic_patch": 6,
#             "repeat_time": 1,
#             "length": len(train_data)
#         },
#         "walk-vlm-val": {
#             "root": "data/walk_vlm/images",
#             "annotation": "data/walk_vlm/walk_val.jsonl",
#             "data_augment": False,
#             "max_dynamic_patch": 6,
#             "repeat_time": 1,
#             "length": len(val_data)
#         }
#     }
#     with open(META_PATH, "w") as f:
#         json.dump(meta_content, f, indent=2)
#     print(f"✅ Updated metadata at {META_PATH}")

# if __name__ == "__main__":
#     args = parse_args()
#     process_data(limit=args.limit, seed=args.seed)

























import os
import json
import argparse
import random
from datasets import load_dataset, Video
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from huggingface_hub import HfFileSystem

# --- Configuration ---
DATASET_REPO = "blind-assist/walk-train"
OUTPUT_DIR = "./data/walk_vlm"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
TRAIN_JSONL_PATH = os.path.join(OUTPUT_DIR, "walk_train.jsonl")
VAL_JSONL_PATH = os.path.join(OUTPUT_DIR, "walk_val.jsonl")
TRAIN_META_PATH = os.path.join(OUTPUT_DIR, "walk_train_meta.json")
VAL_META_PATH = os.path.join(OUTPUT_DIR, "walk_val_meta.json")
FRAMES_PER_VIDEO = 2
VAL_SPLIT_RATIO = 0.1  # 10% for validation

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare WalkVLM Dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process (e.g., 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    return parser.parse_args()

def process_data(limit=None, seed=42, val_ratio=0.1):
    os.makedirs(IMG_DIR, exist_ok=True)
    random.seed(seed)
    
    # Initialize Hugging Face FileSystem to handle hf:// paths
    fs = HfFileSystem()

    print(f"Loading dataset: {DATASET_REPO} (Streaming Mode)...")
    dataset = load_dataset(DATASET_REPO, split="train", streaming=True)
    
    # Disable decoding to get the path/bytes info
    dataset = dataset.cast_column("video", Video(decode=False))

    if limit:
        print(f"⚠️ LIMITING dataset to first {limit} videos.")
        dataset = dataset.take(limit)
        total_to_process = limit
    else:
        total_to_process = None

    formatted_data = []
    print("Processing videos...")
    
    for idx, item in tqdm(enumerate(dataset), total=total_to_process):
        video_path = f"temp_{idx}.mp4"
        try:
            target_text = item.get("alter")
            if not target_text and "json" in item and item["json"]:
                try:
                    data_json = json.loads(item["json"])
                    target_text = data_json.get("alter")
                except:
                    pass

            if not target_text:
                continue

            # --- HANDLE VIDEO DOWNLOAD ---
            video_obj = item.get("video")
            
            if isinstance(video_obj, dict):
                # Case 1: We have raw bytes (small videos sometimes)
                if video_obj.get("bytes"):
                    with open(video_path, "wb") as f:
                        f.write(video_obj["bytes"])
                
                # Case 2: We have a path (hf:// or http:// or local)
                elif video_obj.get("path"):
                    v_path = video_obj["path"]
                    
                    if v_path.startswith("hf://"):
                        # Use HfFileSystem to read the internal HF path
                        with fs.open(v_path, "rb") as r:
                            with open(video_path, "wb") as w:
                                w.write(r.read())
                                
                    elif v_path.startswith("http"):
                        # Standard URL download
                        import requests
                        with open(video_path, 'wb') as f:
                            f.write(requests.get(v_path).content)
                    else:
                        # It's likely a local path (unlikely in streaming, but possible)
                        video_path = v_path
            else:
                continue

            # --- EXTRACT FRAMES ---
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = [int(total_frames * (i + 1) / (FRAMES_PER_VIDEO + 1)) for i in range(FRAMES_PER_VIDEO)]
            
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
                        {"from": "human", "value": "<image>\nAnalyze this scene for navigation hazards."},
                        {"from": "gpt", "value": target_text}
                    ]
                }
                formatted_data.append(entry)

        except Exception as e:
            print(f"Error on index {idx}: {e}")
            continue
        finally:
            # Cleanup temp video file if we created one
            if video_path.startswith("temp_") and os.path.exists(video_path):
                os.remove(video_path)

    # --- SPLIT INTO TRAIN/VAL ---
    random.shuffle(formatted_data)
    val_size = int(len(formatted_data) * val_ratio)
    val_data = formatted_data[:val_size]
    train_data = formatted_data[val_size:]
    
    print(f"📊 Split: {len(train_data)} train, {len(val_data)} validation samples")

    # Save Train JSONL
    with open(TRAIN_JSONL_PATH, "w") as f:
        for entry in train_data:
            json.dump(entry, f)
            f.write('\n')
    
    # Save Validation JSONL
    with open(VAL_JSONL_PATH, "w") as f:
        for entry in val_data:
            json.dump(entry, f)
            f.write('\n')
    
    print(f"✅ Generated {len(train_data)} training samples at {TRAIN_JSONL_PATH}")
    print(f"✅ Generated {len(val_data)} validation samples at {VAL_JSONL_PATH}")

    # Generate SEPARATE metadata files for train and val
    train_meta = {
        "walk-vlm-train": {
            "root": "data/walk_vlm/images",
            "annotation": "data/walk_vlm/walk_train.jsonl",
            "data_augment": False,
            "max_dynamic_patch": 6,
            "repeat_time": 1,
            "length": len(train_data)
        }
    }
    
    val_meta = {
        "walk-vlm-val": {
            "root": "data/walk_vlm/images",
            "annotation": "data/walk_vlm/walk_val.jsonl",
            "data_augment": False,
            "max_dynamic_patch": 6,
            "repeat_time": 1,
            "length": len(val_data)
        }
    }
    
    with open(TRAIN_META_PATH, "w") as f:
        json.dump(train_meta, f, indent=2)
    print(f"✅ Train metadata saved at {TRAIN_META_PATH}")
    
    with open(VAL_META_PATH, "w") as f:
        json.dump(val_meta, f, indent=2)
    print(f"✅ Validation metadata saved at {VAL_META_PATH}")

if __name__ == "__main__":
    args = parse_args()
    process_data(limit=args.limit, seed=args.seed, val_ratio=args.val_ratio)