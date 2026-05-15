import json
import time
import torch
import os
import cv2
import numpy as np
import argparse
from typing import List
from PIL import Image as PILImage
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from datasets import load_dataset, Video

# --- Configuration ---
BASE_MODEL = "OpenGVLab/InternVL3-1B"
LORA_REPO = "blind-assist/internvl3-1b-walk-lora-Epoch3-8500-v1"
HF_DATASET = "blind-assist/walk"

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

def get_video_frames(video_path: str, max_frames: int = 3) -> List[PILImage.Image]:
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

def download_and_merge_lora(model, lora_repo):
    """Download adapter from HF and merge into model state."""
    print(f"🔄 Downloading LoRA adapter from: {lora_repo}")
    
    try:
        adapter_weights_path = hf_hub_download(repo_id=lora_repo, filename="adapter_model.safetensors")
        adapter_config_path = hf_hub_download(repo_id=lora_repo, filename="adapter_config.json")
    except Exception as e:
        print(f"❌ Failed to download adapter: {e}")
        raise
    
    with open(adapter_config_path, 'r') as f:
        config = json.load(f)
    
    adapter_weights = load_file(adapter_weights_path)
    model_state = model.state_dict()
    scaling = config.get('lora_alpha', config['r']) / config['r']
    
    merged_count = 0
    for key in list(adapter_weights.keys()):
        if '.lora_A.' in key:
            lora_b_key = key.replace('.lora_A.', '.lora_B.')
            if lora_b_key in adapter_weights:
                model_key = key.replace('.lora_A.', '.').replace('base_model.model.', '')
                
                if model_key in model_state:
                    device = model_state[model_key].device
                    original_dtype = model_state[model_key].dtype
                    
                    lora_a = adapter_weights[key].float().to(device)
                    lora_b = adapter_weights[lora_b_key].float().to(device)
                    
                    delta = torch.matmul(lora_b, lora_a) * scaling
                    model_state[model_key] = model_state[model_key].float() + delta
                    model_state[model_key] = model_state[model_key].to(original_dtype)
                    merged_count += 1

    model.load_state_dict(model_state)
    print(f"✅ Successfully merged {merged_count} LoRA layers.")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=HF_DATASET)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--lora_repo", type=str, default=LORA_REPO)
    parser.add_argument("--output", type=str, default="./results_epoch3_test")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # 1. Load Model & Tokenizer
    print(f"🔄 Loading base model: {BASE_MODEL}")
    model = AutoModel.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, 
        trust_remote_code=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # 2. Apply Fine-Tuned Weights
    model = download_and_merge_lora(model, args.lora_repo)
    model.eval()

    # 3. Load Dataset (Standard caching logic)
    print(f"📁 Loading Hugging Face dataset: {args.dataset} (Split: {args.split})")
    dataset = load_dataset(args.dataset, split=args.split)
    
    if 'video' in dataset.column_names:
        print("🔧 Bypassing torchcodec and using raw file paths...")
        dataset = dataset.cast_column('video', Video(decode=False))
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"🚀 Processing {len(dataset)} samples")

    prompt = "Given the visual input from the user’s forward perspective, identify the closest immediate obstacle that poses the highest collision risk (especially within approximately 2 meters), and generate exactly one short sentence guiding a visually impaired user by describing its location using clock directions relative to the user (12 o’clock is straight ahead), including relevant details such as size, material, or distance, and giving one clear action to avoid it, prioritizing immediate safety and ignoring less urgent or distant objects, with no extra explanation."

    results_list = []
    
    # 4. Inference Loop
    for idx, sample in enumerate(dataset):
        print(f"\n📦 Processing {idx+1}/{len(dataset)}")
        
        sample_id = f"sample_{idx:04d}"
        
        # Handle Video logic
        if 'video' in sample:
            video_path = sample['video']['path'] if isinstance(sample['video'], dict) else sample['video']
            sample_id = os.path.basename(video_path)
            
            frames = get_video_frames(video_path)
            if not frames: 
                print(f"  ⚠️ OpenCV failed to read frames from {video_path}")
                continue
                
            image = frames[len(frames) // 2]
        else:
            image = sample.get('image') or sample.get('img')

        if not image: continue

        # Prepare input
        pixel_values = process_frame(image).unsqueeze(0).to(model.device, dtype=torch.bfloat16)
        
        t_start = time.time()
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=dict(max_new_tokens=256, do_sample=False)
        )
        elapsed = time.time() - t_start

        results_list.append({
            "sample_id": sample_id,
            "response": response,
            "inference_time_sec": round(elapsed, 2)
        })
        print(f"   ✅ Saved as: {sample_id}")
        print(f"   ✅ Time: {elapsed:.2f}s")
        print(f"   📝 {response}")

    # 5. Export Results
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "finetuned_test_results.json"), "w") as f:
        json.dump(results_list, f, indent=2)
    
    try:
        import pandas as pd
        pd.DataFrame(results_list).to_excel(os.path.join(args.output, "finetuned_test_results.xlsx"), index=False)
        print("\n✅ Results saved to Excel.")
    except ImportError:
        print("\n⚠️ pandas not found. Results saved to JSON.")

if __name__ == "__main__":
    main()