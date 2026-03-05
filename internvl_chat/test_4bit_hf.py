import json
import time
import torch
import glob
import os
import cv2
import numpy as np
from typing import List
from PIL import Image as PILImage
import transformers
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# --- MONKEY PATCH 1: 4-bit Quantization Tied Weights Fix ---
try:
    import transformers.quantizers.base
    original_get_keys = transformers.quantizers.base.get_keys_to_not_convert
    def patched_get_keys(model, *args, **kwargs):
        if not hasattr(model, 'all_tied_weights_keys'):
            model.all_tied_weights_keys = {}
        return original_get_keys(model, *args, **kwargs)
    transformers.quantizers.base.get_keys_to_not_convert = patched_get_keys
except ImportError:
    pass

# --- MONKEY PATCH 3: Non-Quantized Memory Allocation Fix ---
# Fixes the missing attribute error when loading in full 16-bit without 4-bit
try:
    import transformers.modeling_utils
    original_get_total_byte_count = transformers.modeling_utils.get_total_byte_count
    def patched_get_total_byte_count(model, *args, **kwargs):
        if not hasattr(model, 'all_tied_weights_keys'):
            model.all_tied_weights_keys = {}
        return original_get_total_byte_count(model, *args, **kwargs)
    transformers.modeling_utils.get_total_byte_count = patched_get_total_byte_count
except ImportError:
    pass
# -----------------------------------------------------------

try:
    from peft import PeftModel
except ImportError:
    print("❌ PEFT not found. Please run: pip install peft")
    raise

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# --- Configuration ---
BASE_MODEL = "OpenGVLab/InternVL3-2B" 
LORA_REPO = "blind-assist/internvl3-2b-walk-lora-Epoch3-8500-v2"

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

def load_patched_base_model(base_model_name, use_4bit=False):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    ) if use_4bit else None

    # --- MONKEY PATCH 2: InternVL Meta-Device Bug Fix ---
    original_linspace = torch.linspace
    def safe_linspace(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return original_linspace(*args, **kwargs)
    
    torch.linspace = safe_linspace
    # ----------------------------------------------------

    try:
        model = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            device_map={"": 0}, 
            quantization_config=quant_config,
            low_cpu_mem_usage=True
        )
    finally:
        torch.linspace = original_linspace

    return model

def load_model_and_lora(base_model_name, lora_repo, use_4bit=False):
    print(f"🔄 Loading base model: {base_model_name}")
    model = load_patched_base_model(base_model_name, use_4bit)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print(f"🔄 Attaching LoRA adapter: {lora_repo}")
    model = PeftModel.from_pretrained(model, lora_repo)

    if not use_4bit:
        print("📊 Merging LoRA weights directly into the base model for maximum speed...")
        model = model.merge_and_unload()
    else:
        print("📊 4-bit mode: LoRA active as adapter.")

    model.eval()
    print("✅ Model ready!")
    return model, tokenizer

def test_single_image(model, tokenizer, image_path, prompt):
    image = PILImage.open(image_path).convert('RGB')
    pixel_values = process_frame(image, input_size=448)
    pixel_values = pixel_values.unsqueeze(0).to(model.device, dtype=torch.bfloat16)
    
    t_start = time.time()
    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            # OPTIMIZED: Reduced tokens from 256 to 50 for faster generation
            generation_config=dict(max_new_tokens=50, do_sample=False)
        )
    t_end = time.time()
    
    return response, t_end - t_start

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test fine-tuned InternVL3")
    parser.add_argument("--input", type=str, default="./my_videos")
    parser.add_argument("--output", type=str, default="./inference_results_hf")
    parser.add_argument("--base", action="store_true", help="Use base model only")
    parser.add_argument("--lora_repo", type=str, default=LORA_REPO)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--use_4bit", action="store_true")
    args = parser.parse_args()

    if args.base:
        print(f"🔄 Loading BASE model ONLY: {BASE_MODEL}")
        model = load_patched_base_model(BASE_MODEL, use_4bit=args.use_4bit)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model.eval()
        model_name = "base"
        print("✅ Base Model ready!")
    else:
        model, tokenizer = load_model_and_lora(BASE_MODEL, args.lora_repo, use_4bit=args.use_4bit)
        model_name = "finetuned_hf"

    prompt = ("Given the visual input from the user’s forward perspective, identify the closest immediate "
              "obstacle that poses the highest collision risk (especially within approximately 2 meters), "
              "and generate exactly one short sentence guiding a visually impaired user by describing its "
              "location using clock directions relative to the user (12 o’clock is straight ahead), "
              "including relevant details such as size, material, or distance, and giving one clear action "
              "to avoid it, prioritizing immediate safety and ignoring less urgent or distant objects, "
              "with no extra explanation.")

    if args.image:
        print(f"\n📷 Testing image: {args.image}")
        response, elapsed = test_single_image(model, tokenizer, args.image, prompt)
        print(f"⏱️ Time: {elapsed:.2f}s\n📝 Response: {response}")
        return

    print(f"\n📁 Input folder: {args.input}")
    video_paths = sorted(glob.glob(os.path.join(args.input, "*.*")))
    video_paths = [p for p in video_paths if p.lower().endswith(('.mp4', '.avi', '.jpg', '.png', '.jpeg'))]

    if not video_paths:
        print(f"❌ No videos/images found in {args.input}")
        return

    if args.max_videos:
        video_paths = video_paths[:args.max_videos]

    print(f"🚀 Processing {len(video_paths)} files")
    results_list = []

    for idx, video_path in enumerate(video_paths):
        filename = os.path.basename(video_path)
        print(f"\n📦 Processing {idx+1}/{len(video_paths)}: {filename}")
        
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
        
        start = time.time()
        with torch.no_grad():
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                # OPTIMIZED: Reduced tokens from 256 to 50 for faster generation
                generation_config=dict(max_new_tokens=50, do_sample=False)
            )
        elapsed = time.time() - start
        
        results_list.append({
            "filename": filename,
            "response": response,
            "inference_time_sec": round(elapsed, 2),
            "model": model_name,
            "prompt": prompt
        })
        print(f"   ✅ Time: {elapsed:.2f}s")
        print(f"   📝 {response[:200]}...")

    os.makedirs(args.output, exist_ok=True)
    
    json_file = os.path.join(args.output, f"{model_name}_results.json")
    with open(json_file, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"\n✅ JSON saved to {json_file}")
    
    try:
        import pandas as pd
        df = pd.DataFrame(results_list)
        excel_file = os.path.join(args.output, f"{model_name}_results.xlsx")
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✅ Excel saved to {excel_file}")
    except ImportError:
        csv_file = os.path.join(args.output, f"{model_name}_results.csv")
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("filename,response,inference_time_sec,model,prompt\n")
            for r in results_list:
                resp_escaped = r["response"].replace('"', '""')
                f.write(f'"{r["filename"]}","{resp_escaped}",{r["inference_time_sec"]},"{r["model"]}","{r["prompt"]}"\n')
        print(f"✅ CSV saved to {csv_file}")

if __name__ == "__main__":
    main()