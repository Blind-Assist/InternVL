# import json
# import time
# import torch
# import glob
# import os
# import cv2
# import numpy as np
# import argparse
# from typing import List
# from PIL import Image as PILImage
# from transformers import AutoModel, AutoTokenizer
# from safetensors.torch import load_file
# import torchvision.transforms as T
# from torchvision.transforms.functional import InterpolationMode
# from datasets import load_dataset, Video  # <-- ADDED Video import

# # Your trained model path (local)
# TRAINED_MODEL_PATH = "work_dirs/internvl3_2b_walk_lora" 
# BASE_MODEL = "OpenGVLab/InternVL3-2B"     

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)

# def build_transform(input_size):
#     return T.Compose([
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
#     ])

# def process_frame(frame: PILImage.Image, input_size=448):
#     transform = build_transform(input_size)
#     return transform(frame)

# def get_video_frames(video_path: str, max_frames: int = 8) -> List[PILImage.Image]:
#     """Extracts frames from a video file."""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return []
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
#     frames = []
#     for i in indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(PILImage.fromarray(frame))
#     cap.release()
#     return frames

# def load_and_merge_lora_weights(model, checkpoint_path):
#     print(f"🔄 Loading checkpoint weights from: {checkpoint_path}")
#     safetensor_files = sorted(glob.glob(os.path.join(checkpoint_path, "model*.safetensors")))
#     if not safetensor_files:
#         raise FileNotFoundError(f"No safetensor files found in {checkpoint_path}")
    
#     all_weights = {}
#     for sf_file in safetensor_files:
#         print(f"   Loading: {os.path.basename(sf_file)}")
#         weights = load_file(sf_file)
#         all_weights.update(weights)
#     print(f"   📊 Total checkpoint tensors: {len(all_weights)}")
    
#     base_weights, lora_a_weights, lora_b_weights, other_weights = {}, {}, {}, {}
#     for key, value in all_weights.items():
#         if '.lora_A.' in key: lora_a_weights[key] = value
#         elif '.lora_B.' in key: lora_b_weights[key] = value
#         elif '.base_layer.' in key: base_weights[key] = value
#         else: other_weights[key] = value
            
#     model_state = model.state_dict()
#     merged_count = 0
#     scaling = 1.0 
#     new_state_dict = {}
    
#     for base_key, base_value in base_weights.items():
#         lora_a_key = base_key.replace('.base_layer.', '.lora_A.default.')
#         lora_b_key = base_key.replace('.base_layer.', '.lora_B.default.')
#         model_key = base_key.replace('base_model.model.', '').replace('.base_layer', '')
        
#         if lora_a_key in lora_a_weights and lora_b_key in lora_b_weights:
#             lora_a = lora_a_weights[lora_a_key].float()
#             lora_b = lora_b_weights[lora_b_key].float()
#             base = base_value.float()
#             delta = torch.matmul(lora_b, lora_a) * scaling
#             merged = base + delta
#             new_state_dict[model_key] = merged.to(base_value.dtype)
#             merged_count += 1
#         else:
#             new_state_dict[model_key] = base_value
            
#     for key, value in other_weights.items():
#         model_key = key.replace('base_model.model.', '')
#         new_state_dict[model_key] = value
        
#     loaded_keys = []
#     for key, value in new_state_dict.items():
#         if key in model_state:
#             if model_state[key].shape == value.shape:
#                 model_state[key] = value
#                 loaded_keys.append(key)
                
#     model.load_state_dict(model_state)
#     print(f"\n   ✅ Successfully merged {merged_count} LoRA layers")
#     print(f"   ✅ Total weights loaded: {len(loaded_keys)}")
#     return model

# def load_model_with_merged_lora(checkpoint_path, base_model_name):
#     print(f"🔄 Loading base model: {base_model_name}")
#     model = AutoModel.from_pretrained(
#         base_model_name,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
#     model = load_and_merge_lora_weights(model, checkpoint_path)
#     model.eval()
#     print("✅ Model loaded with MERGED LoRA fine-tuned weights!")
#     return model, tokenizer

# def load_base_model_only(base_model_name):
#     print(f"🔄 Loading base model: {base_model_name}")
#     model = AutoModel.from_pretrained(
#         base_model_name,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
#     model.eval()
#     print("✅ Base model loaded!")
#     return model, tokenizer

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, default="blind-assist/walk", help="Hugging Face dataset name")
#     parser.add_argument("--split", type=str, default="test", help="Dataset split (e.g., test, train)")
#     parser.add_argument("--output", type=str, default="./inference_results_hf_dataset")
#     parser.add_argument("--base", action="store_true", help="Use base model only (no fine-tuning)")
#     parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
#     parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
#     args = parser.parse_args()

#     if args.base:
#         model, tokenizer = load_base_model_only(BASE_MODEL)
#         model_name = "base"
#     else:
#         checkpoint_path = args.checkpoint if args.checkpoint else TRAINED_MODEL_PATH
#         model, tokenizer = load_model_with_merged_lora(checkpoint_path, BASE_MODEL)
#         model_name = "finetuned"

#     print(f"\n📁 Loading Hugging Face dataset: {args.dataset} (Split: {args.split})")
    
#     try:
#         dataset = load_dataset(args.dataset, split=args.split)
        
#         # --- THE MAGIC FIX ---
#         # Disable HuggingFace's automatic video decoding to prevent torchcodec crash
#         if 'video' in dataset.column_names:
#             print("🔧 Bypassing torchcodec and using raw file paths...")
#             dataset = dataset.cast_column('video', Video(decode=False))
            
#     except Exception as e:
#         print(f"❌ Failed to load dataset. Error: {e}")
#         return

#     if args.max_samples:
#         dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
#     print(f"🚀 Processing {len(dataset)} samples")
    
#     prompt = "Given the visual input from the user’s forward perspective, generate exactly one short sentence to guide a visually impaired user by identifying critical obstacles or landmarks, describing their locations using clock directions relative to the user (12 o’clock is straight ahead), including relevant details such as size, material, or distance, and giving one clear action, while prioritizing immediate safety and avoiding any extra explanation."

#     results_list = []
#     results = {"model": model_name, "predictions": {}}

#     for idx, sample in enumerate(dataset):
#         print(f"\n📦 Processing {idx+1}/{len(dataset)}")
        
#         # Extract the video path directly and let OpenCV do the heavy lifting
#         if 'video' in sample:
#             video_data = sample['video']
#             video_path = video_data['path'] if isinstance(video_data, dict) else video_data
            
#             frames = get_video_frames(video_path)
#             if not frames:
#                 print(f"⚠️ CV2 failed to extract frames from {video_path}")
#                 continue
#             image = frames[len(frames) // 2]
            
#         elif 'image' in sample:
#             image = sample['image']
#         elif 'img' in sample:
#             image = sample['img']
#         else:
#             print(f"⚠️ Could not find a video/image column. Available columns: {sample.keys()}")
#             continue
            
#         if not isinstance(image, PILImage.Image):
#             if isinstance(image, np.ndarray):
#                 image = PILImage.fromarray(image)
#             else:
#                 print(f"⚠️ Sample {idx} frame is not a PIL Image. Skipping.")
#                 continue
            
#         pixel_values = process_frame(image, input_size=448)
#         pixel_values = pixel_values.unsqueeze(0).to(model.device, dtype=torch.bfloat16)
        
#         t_start = time.time()
#         response = model.chat(
#             tokenizer=tokenizer,
#             pixel_values=pixel_values,
#             question=prompt,
#             generation_config=dict(max_new_tokens=256, do_sample=False)
#         )
#         t_end = time.time()
#         elapsed = t_end - t_start
        
#         sample_id = str(sample.get('id', sample.get('filename', f"sample_{idx:04d}")))
#         results["predictions"][sample_id] = response
        
#         results_list.append({
#             "sample_id": sample_id,
#             "response": response,
#             "inference_time_sec": round(elapsed, 2),
#             "model": model_name,
#             "prompt": prompt
#         })
        
#         print(f"   ✅ Time: {elapsed:.2f}s")
#         print(f"   📝 {response[:200]}...")

#     os.makedirs(args.output, exist_ok=True)
#     json_file = os.path.join(args.output, f"{model_name}_hf_results.json")
#     with open(json_file, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\n✅ JSON saved to {json_file}")
    
#     try:
#         import pandas as pd
#         df = pd.DataFrame(results_list)
#         excel_file = os.path.join(args.output, f"{model_name}_hf_results.xlsx")
#         df.to_excel(excel_file, index=False, engine='openpyxl')
#         print(f"✅ Excel saved to {excel_file}")
#     except ImportError:
#         csv_file = os.path.join(args.output, f"{model_name}_hf_results.csv")
#         with open(csv_file, "w", encoding="utf-8") as f:
#             f.write("sample_id,response,inference_time_sec,model,prompt\n")
#             for r in results_list:
#                 response_escaped = r["response"].replace('"', '""')
#                 f.write(f'"{r["sample_id"]}","{response_escaped}",{r["inference_time_sec"]},"{r["model"]}","{r["prompt"]}"\n')
#         print(f"✅ CSV saved to {csv_file}")

# if __name__ == "__main__":
#     main()























import json
import time
import torch
import glob
import os
import cv2
import numpy as np
import argparse
from typing import List
from PIL import Image as PILImage
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from datasets import load_dataset, Video

# Your trained model path (local)
TRAINED_MODEL_PATH = "work_dirs/internvl3_2b_walk_lora" 
BASE_MODEL = "OpenGVLab/InternVL3-2B"     

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def process_frame(frame: PILImage.Image, input_size=448):
    transform = build_transform(input_size)
    return transform(frame)

def get_video_frames(video_path: str, max_frames: int = 8) -> List[PILImage.Image]:
    """Extracts frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(PILImage.fromarray(frame))
    cap.release()
    return frames

def load_and_merge_lora_weights(model, checkpoint_path):
    print(f"🔄 Loading checkpoint weights from: {checkpoint_path}")
    safetensor_files = sorted(glob.glob(os.path.join(checkpoint_path, "model*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in {checkpoint_path}")
    
    all_weights = {}
    for sf_file in safetensor_files:
        print(f"   Loading: {os.path.basename(sf_file)}")
        weights = load_file(sf_file)
        all_weights.update(weights)
    print(f"   📊 Total checkpoint tensors: {len(all_weights)}")
    
    base_weights, lora_a_weights, lora_b_weights, other_weights = {}, {}, {}, {}
    for key, value in all_weights.items():
        if '.lora_A.' in key: lora_a_weights[key] = value
        elif '.lora_B.' in key: lora_b_weights[key] = value
        elif '.base_layer.' in key: base_weights[key] = value
        else: other_weights[key] = value
            
    model_state = model.state_dict()
    merged_count = 0
    scaling = 1.0 
    new_state_dict = {}
    
    for base_key, base_value in base_weights.items():
        lora_a_key = base_key.replace('.base_layer.', '.lora_A.default.')
        lora_b_key = base_key.replace('.base_layer.', '.lora_B.default.')
        model_key = base_key.replace('base_model.model.', '').replace('.base_layer', '')
        
        if lora_a_key in lora_a_weights and lora_b_key in lora_b_weights:
            lora_a = lora_a_weights[lora_a_key].float()
            lora_b = lora_b_weights[lora_b_key].float()
            base = base_value.float()
            delta = torch.matmul(lora_b, lora_a) * scaling
            merged = base + delta
            new_state_dict[model_key] = merged.to(base_value.dtype)
            merged_count += 1
        else:
            new_state_dict[model_key] = base_value
            
    for key, value in other_weights.items():
        model_key = key.replace('base_model.model.', '')
        new_state_dict[model_key] = value
        
    loaded_keys = []
    for key, value in new_state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                model_state[key] = value
                loaded_keys.append(key)
                
    model.load_state_dict(model_state)
    print(f"\n   ✅ Successfully merged {merged_count} LoRA layers")
    print(f"   ✅ Total weights loaded: {len(loaded_keys)}")
    return model

def load_model_with_merged_lora(checkpoint_path, base_model_name):
    print(f"🔄 Loading base model: {base_model_name}")
    model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = load_and_merge_lora_weights(model, checkpoint_path)
    model.eval()
    print("✅ Model loaded with MERGED LoRA fine-tuned weights!")
    return model, tokenizer

def load_base_model_only(base_model_name):
    print(f"🔄 Loading base model: {base_model_name}")
    model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model.eval()
    print("✅ Base model loaded!")
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="blind-assist/walk", help="Hugging Face dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (e.g., test, train)")
    parser.add_argument("--output", type=str, default="./inference_results_hf_dataset_withname")
    parser.add_argument("--base", action="store_true", help="Use base model only (no fine-tuning)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    args = parser.parse_args()

    if args.base:
        model, tokenizer = load_base_model_only(BASE_MODEL)
        model_name = "base"
    else:
        checkpoint_path = args.checkpoint if args.checkpoint else TRAINED_MODEL_PATH
        model, tokenizer = load_model_with_merged_lora(checkpoint_path, BASE_MODEL)
        model_name = "finetuned"

    print(f"\n📁 Loading Hugging Face dataset: {args.dataset} (Split: {args.split})")
    
    try:
        dataset = load_dataset(args.dataset, split=args.split)
        
        # Disable HuggingFace's automatic video decoding to prevent torchcodec crash
        if 'video' in dataset.column_names:
            print("🔧 Bypassing torchcodec and using raw file paths...")
            dataset = dataset.cast_column('video', Video(decode=False))
            
    except Exception as e:
        print(f"❌ Failed to load dataset. Error: {e}")
        return

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    print(f"🚀 Processing {len(dataset)} samples")
    
    prompt = "Given the visual input from the user’s forward perspective, generate exactly one short sentence to guide a visually impaired user by identifying critical obstacles or landmarks, describing their locations using clock directions relative to the user (12 o’clock is straight ahead), including relevant details such as size, material, or distance, and giving one clear action, while prioritizing immediate safety and avoiding any extra explanation."

    results_list = []
    results = {"model": model_name, "predictions": {}}

    for idx, sample in enumerate(dataset):
        print(f"\n📦 Processing {idx+1}/{len(dataset)}")
        
        # Default fallback
        sample_id = f"sample_{idx:04d}"
        
        if 'video' in sample:
            video_data = sample['video']
            video_path = video_data['path'] if isinstance(video_data, dict) else video_data
            
            # --- AGGRESSIVELY FORCE THE .MP4 NAME ---
            if isinstance(video_path, str):
                sample_id = os.path.basename(video_path)
            
            frames = get_video_frames(video_path)
            if not frames:
                print(f"⚠️ CV2 failed to extract frames from {video_path}")
                continue
            image = frames[len(frames) // 2]
            
        elif 'image' in sample:
            image = sample['image']
        elif 'img' in sample:
            image = sample['img']
        else:
            print(f"⚠️ Could not find a video/image column. Available columns: {sample.keys()}")
            continue
            
        if not isinstance(image, PILImage.Image):
            if isinstance(image, np.ndarray):
                image = PILImage.fromarray(image)
            else:
                print(f"⚠️ Sample {idx} frame is not a PIL Image. Skipping.")
                continue
            
        pixel_values = process_frame(image, input_size=448)
        pixel_values = pixel_values.unsqueeze(0).to(model.device, dtype=torch.bfloat16)
        
        t_start = time.time()
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=dict(max_new_tokens=256, do_sample=False)
        )
        t_end = time.time()
        elapsed = t_end - t_start
        
        # Save using the exact filename
        results["predictions"][sample_id] = response
        
        results_list.append({
            "sample_id": sample_id,
            "response": response,
            "inference_time_sec": round(elapsed, 2),
            "model": model_name,
            "prompt": prompt
        })
        
        print(f"   ✅ Saved as: {sample_id}")
        print(f"   ✅ Time: {elapsed:.2f}s")
        print(f"   📝 {response[:200]}...")

    os.makedirs(args.output, exist_ok=True)
    json_file = os.path.join(args.output, f"{model_name}_hf_results.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ JSON saved to {json_file}")
    
    try:
        import pandas as pd
        df = pd.DataFrame(results_list)
        excel_file = os.path.join(args.output, f"{model_name}_hf_results.xlsx")
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✅ Excel saved to {excel_file}")
    except ImportError:
        csv_file = os.path.join(args.output, f"{model_name}_hf_results.csv")
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("sample_id,response,inference_time_sec,model,prompt\n")
            for r in results_list:
                response_escaped = r["response"].replace('"', '""')
                f.write(f'"{r["sample_id"]}","{response_escaped}",{r["inference_time_sec"]},"{r["model"]}","{r["prompt"]}"\n')
        print(f"✅ CSV saved to {csv_file}")

if __name__ == "__main__":
    main()