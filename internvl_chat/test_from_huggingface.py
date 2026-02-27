"""
Test fine-tuned InternVL model from HuggingFace.
Downloads LoRA adapter and merges with base model.
"""
import json
import time
import torch
import glob
import os
import cv2
import numpy as np
from typing import List
from PIL import Image as PILImage
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# --- Configuration ---
# BASE_MODEL = "OpenGVLab/InternVL2_5-4B"
BASE_MODEL = "OpenGVLab/InternVL3-2B"     # CHANGED
LORA_REPO = "blind-assist/internvl3-2b-walk-lora-Epoch3-8500-v2"  # Your HuggingFace repo

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
    """Download adapter from HuggingFace and merge ALL weights into model."""
    print(f"🔄 Downloading LoRA adapter from: {lora_repo}")
    
    try:
        adapter_weights_path = hf_hub_download(repo_id=lora_repo, filename="adapter_model.safetensors")
        adapter_config_path = hf_hub_download(repo_id=lora_repo, filename="adapter_config.json")
    except Exception as e:
        print(f"❌ Failed to download adapter: {e}")
        raise
    
    with open(adapter_config_path, 'r') as f:
        config = json.load(f)
    
    print(f"   📊 LoRA rank: {config['r']}")
    print(f"   📊 LoRA alpha: {config['lora_alpha']}")
    print(f"   📊 Target modules: {config['target_modules']}")
    
    adapter_weights = load_file(adapter_weights_path)
    print(f"   📊 Adapter tensors loaded: {len(adapter_weights)}")
    
    model_state = model.state_dict()
    scaling = config.get('lora_alpha', config['r']) / config['r']
    print(f"   📊 Scaling factor: {scaling}")
    
    merged_lora_count = 0
    loaded_other_count = 0
    
    # First pass: Merge LoRA weights
    for key in list(adapter_weights.keys()):
        if '.lora_A.' in key:
            lora_b_key = key.replace('.lora_A.', '.lora_B.')
            
            if lora_b_key in adapter_weights:
                lora_a = adapter_weights[key].float()
                lora_b = adapter_weights[lora_b_key].float()
                
                # Convert key to model format
                model_key = key.replace('.lora_A.', '.').replace('base_model.model.', '')
                
                if model_key in model_state:
                    device = model_state[model_key].device
                    original_dtype = model_state[model_key].dtype
                    
                    lora_a = lora_a.to(device)
                    lora_b = lora_b.to(device)
                    
                    delta = torch.matmul(lora_b, lora_a) * scaling
                    model_state[model_key] = model_state[model_key].float() + delta
                    model_state[model_key] = model_state[model_key].to(original_dtype)
                    merged_lora_count += 1
    
    # Second pass: Load other fine-tuned weights (non-LoRA)
    for key, value in adapter_weights.items():
        # Skip LoRA weights (already handled)
        if '.lora_A.' in key or '.lora_B.' in key:
            continue
        
        # Convert key format
        model_key = key.replace('base_model.model.', '')
        
        # Try different key formats
        possible_keys = [
            model_key,
            key,  # Original key
            model_key.replace('language_model.', ''),
        ]
        
        for try_key in possible_keys:
            if try_key in model_state:
                if model_state[try_key].shape == value.shape:
                    device = model_state[try_key].device
                    dtype = model_state[try_key].dtype
                    model_state[try_key] = value.to(device=device, dtype=dtype)
                    loaded_other_count += 1
                    break
    
    model.load_state_dict(model_state)
    print(f"   ✅ Successfully merged {merged_lora_count} LoRA layers")
    print(f"   ✅ Loaded {loaded_other_count} other fine-tuned weights")
    
    return model


def load_finetuned_model_from_hf(base_model_name, lora_repo):
    """Load base model and apply LoRA adapter from HuggingFace."""
    print(f"🔄 Loading base model: {base_model_name}")
    
    model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, 
        trust_remote_code=True,
    )
    
    # Download and merge LoRA weights from HuggingFace
    model = download_and_merge_lora(model, lora_repo)
    
    model.eval()
    print("✅ Fine-tuned model ready!")
    return model, tokenizer

def load_base_model_only(base_model_name):
    """Load just the base model without fine-tuning (for comparison)."""
    print(f"🔄 Loading base model: {base_model_name}")
    
    model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, 
        trust_remote_code=True,
    )
    
    model.eval()
    print("✅ Base model loaded!")
    return model, tokenizer

def test_single_image(model, tokenizer, image_path, prompt):
    """Test with a single image."""
    image = PILImage.open(image_path).convert('RGB')
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
    
    return response, t_end - t_start

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test fine-tuned model from HuggingFace")
    parser.add_argument("--input", type=str, default="./my_videos", help="Input folder with videos/images")
    parser.add_argument("--output", type=str, default="./inference_results_hf", help="Output folder")
    parser.add_argument("--base", action="store_true", help="Use base model only (no LoRA)")
    parser.add_argument("--lora_repo", type=str, default=LORA_REPO, help="HuggingFace LoRA repo")
    parser.add_argument("--max_videos", type=int, default=None, help="Max videos to process")
    parser.add_argument("--image", type=str, default=None, help="Test single image instead of videos")
    args = parser.parse_args()

    # Load model
    if args.base:
        model, tokenizer = load_base_model_only(BASE_MODEL)
        model_name = "base"
    else:
        model, tokenizer = load_finetuned_model_from_hf(BASE_MODEL, args.lora_repo)
        model_name = "finetuned_hf"

    # prompt = "Given the visual input from the user's forward perspective, generate exactly one short sentence to guide a visually impaired user by identifying critical obstacles or landmarks, describing their locations using clock directions relative to the user (12 o'clock is straight ahead), including relevant details such as size, material, or distance, and giving one clear action, while prioritizing immediate safety and avoiding any extra explanation."
    prompt = "Given the visual input from the user’s forward perspective, identify the closest immediate obstacle that poses the highest collision risk (especially within approximately 2 meters), and generate exactly one short sentence guiding a visually impaired user by describing its location using clock directions relative to the user (12 o’clock is straight ahead), including relevant details such as size, material, or distance, and giving one clear action to avoid it, prioritizing immediate safety and ignoring less urgent or distant objects, with no extra explanation."
    # Test single image mode
    if args.image:
        print(f"\n📷 Testing single image: {args.image}")
        response, elapsed = test_single_image(model, tokenizer, args.image, prompt)
        print(f"⏱️ Time: {elapsed:.2f}s")
        print(f"📝 Response:\n{response}")
        return

    # Test videos mode
    print(f"\n📁 Input folder: {args.input}")
    
    video_paths = sorted(
        glob.glob(os.path.join(args.input, "*.mp4")) + 
        glob.glob(os.path.join(args.input, "*.avi")) +
        glob.glob(os.path.join(args.input, "*.jpg")) +
        glob.glob(os.path.join(args.input, "*.png"))
    )
    
    if not video_paths:
        print(f"❌ No videos/images found in {args.input}")
        return

    if args.max_videos:
        video_paths = video_paths[:args.max_videos]
    
    print(f"🚀 Processing {len(video_paths)} files")
    
    # Store results for Excel
    results_list = []
    results = {"model": model_name, "lora_repo": args.lora_repo if not args.base else None, "predictions": {}}

    for idx, video_path in enumerate(video_paths):
        filename = os.path.basename(video_path)
        print(f"\n📦 Processing {idx+1}/{len(video_paths)}: {filename}")
        
        # Handle images vs videos
        if video_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            frames = [PILImage.open(video_path)]
        else:
            frames = get_video_frames(video_path)
        
        if not frames:
            print(f"   ⚠️ Could not read frames")
            continue
        
        middle_frame = frames[len(frames) // 2]
        pixel_values = process_frame(middle_frame, input_size=448)
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
        
        results["predictions"][filename] = response
        
        # Add to results list for Excel
        results_list.append({
            "filename": filename,
            "response": response,
            "inference_time_sec": round(elapsed, 2),
            "model": model_name,
            "prompt": prompt
        })
        
        print(f"   ✅ Time: {elapsed:.2f}s")
        print(f"   📝 {response[:200]}...")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    # Save JSON
    json_file = os.path.join(args.output, f"{model_name}_results.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ JSON saved to {json_file}")
    
    # Save Excel
    try:
        import pandas as pd
        df = pd.DataFrame(results_list)
        excel_file = os.path.join(args.output, f"{model_name}_results.xlsx")
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✅ Excel saved to {excel_file}")
    except ImportError:
        print("⚠️ pandas or openpyxl not installed. Install with: pip install pandas openpyxl")
        # Fallback to CSV
        csv_file = os.path.join(args.output, f"{model_name}_results.csv")
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("filename,response,inference_time_sec,model,prompt\n")
            for r in results_list:
                # Escape quotes in response
                response_escaped = r["response"].replace('"', '""')
                f.write(f'"{r["filename"]}","{response_escaped}",{r["inference_time_sec"]},"{r["model"]}","{r["prompt"]}"\n')
        print(f"✅ CSV saved to {csv_file}")

if __name__ == "__main__":
    main()









# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description="Test fine-tuned model from HuggingFace")
#     parser.add_argument("--input", type=str, default="./my_videos", help="Input folder with videos/images")
#     parser.add_argument("--output", type=str, default="./inference_results_hf", help="Output folder")
#     parser.add_argument("--base", action="store_true", help="Use base model only (no LoRA)")
#     parser.add_argument("--lora_repo", type=str, default=LORA_REPO, help="HuggingFace LoRA repo")
#     parser.add_argument("--max_videos", type=int, default=None, help="Max videos to process")
#     parser.add_argument("--image", type=str, default=None, help="Test single image instead of videos")
#     args = parser.parse_args()

#     # Load model
#     if args.base:
#         model, tokenizer = load_base_model_only(BASE_MODEL)
#         model_name = "base"
#     else:
#         model, tokenizer = load_finetuned_model_from_hf(BASE_MODEL, args.lora_repo)
#         model_name = "finetuned_hf"

#     prompt = "You are an assistive navigation system for a visually impaired user. Analyze this scene and identify all immediate, high-risk obstructions. State each obstruction's location using the 12-hour clock face. Generate a single, actionable safety alert."

#     # Test single image mode
#     if args.image:
#         print(f"\n📷 Testing single image: {args.image}")
#         response, elapsed = test_single_image(model, tokenizer, args.image, prompt)
#         print(f"⏱️ Time: {elapsed:.2f}s")
#         print(f"📝 Response:\n{response}")
#         return

#     # Test videos mode
#     print(f"\n📁 Input folder: {args.input}")
    
#     video_paths = sorted(
#         glob.glob(os.path.join(args.input, "*.mp4")) + 
#         glob.glob(os.path.join(args.input, "*.avi")) +
#         glob.glob(os.path.join(args.input, "*.jpg")) +
#         glob.glob(os.path.join(args.input, "*.png"))
#     )
    
#     if not video_paths:
#         print(f"❌ No videos/images found in {args.input}")
#         return

#     if args.max_videos:
#         video_paths = video_paths[:args.max_videos]
    
#     print(f"🚀 Processing {len(video_paths)} files")
    
#     results = {"model": model_name, "lora_repo": args.lora_repo if not args.base else None, "predictions": {}}

#     for idx, video_path in enumerate(video_paths):
#         filename = os.path.basename(video_path)
#         print(f"\n📦 Processing {idx+1}/{len(video_paths)}: {filename}")
        
#         # Handle images vs videos
#         if video_path.lower().endswith(('.jpg', '.png', '.jpeg')):
#             frames = [PILImage.open(video_path)]
#         else:
#             frames = get_video_frames(video_path)
        
#         if not frames:
#             print(f"   ⚠️ Could not read frames")
#             continue
        
#         middle_frame = frames[len(frames) // 2]
#         pixel_values = process_frame(middle_frame, input_size=448)
#         pixel_values = pixel_values.unsqueeze(0).to(model.device, dtype=torch.bfloat16)
        
#         t_start = time.time()
#         response = model.chat(
#             tokenizer=tokenizer,
#             pixel_values=pixel_values,
#             question=prompt,
#             generation_config=dict(max_new_tokens=256, do_sample=False)
#         )
#         t_end = time.time()
        
#         results["predictions"][filename] = response
#         print(f"   ✅ Time: {t_end - t_start:.2f}s")
#         print(f"   📝 {response[:200]}...")

#     # Save results
#     os.makedirs(args.output, exist_ok=True)
#     output_file = os.path.join(args.output, f"{model_name}_results.json")
#     with open(output_file, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\n✅ Saved to {output_file}")

# if __name__ == "__main__":
#     main()