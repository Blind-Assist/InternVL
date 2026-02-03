# Cell: Test your TRAINED model (it's a full fine-tuned model, not LoRA)

import json
import time
import torch
import glob
import os
import cv2
import numpy as np
from typing import List
from PIL import Image as PILImage
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Your trained model path (local)
TRAINED_MODEL_PATH = "work_dirs/internvl2_5_4b_walk_lora"
BASE_MODEL = "OpenGVLab/InternVL2_5-4B"

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

def load_model(model_path, is_local=True):
    """Load model from local path or HuggingFace."""
    print(f"🔄 Loading model from: {model_path}")
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=is_local
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=is_local
    )
    
    model.eval()
    print("✅ Model loaded!")
    return model, tokenizer

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./my_videos")
    parser.add_argument("--output", type=str, default="./inference_results")
    parser.add_argument("--base", action="store_true", help="Use base model")
    parser.add_argument("--model_path", type=str, default=None, help="Custom model path")
    args = parser.parse_args()

    # Determine which model to load
    if args.base:
        model_path = BASE_MODEL
        is_local = False
        model_name = "base"
    elif args.model_path:
        model_path = args.model_path
        is_local = os.path.exists(model_path)
        model_name = "custom"
    else:
        model_path = TRAINED_MODEL_PATH
        is_local = True
        model_name = "trained"

    print(f"📁 Input folder: {args.input}")
    
    video_paths = sorted(glob.glob(os.path.join(args.input, "*.mp4")) + 
                         glob.glob(os.path.join(args.input, "*.avi")))
    
    if not video_paths:
        print(f"❌ No videos found")
        return

    print(f"🚀 Found {len(video_paths)} videos")
    model, tokenizer = load_model(model_path, is_local)
    
    prompt = "You are an assistive navigation system for a visually impaired user. Analyze this scene and identify all immediate, high-risk obstructions. State each obstruction's location using the 12-hour clock face. Generate a single, actionable safety alert."
    
    results = {"model": model_name, "model_path": model_path, "predictions": {}}

    for idx, video_path in enumerate(video_paths[:5]):  # Test first 5
        filename = os.path.basename(video_path)
        print(f"\n📦 Processing {idx+1}: {filename}")
        
        frames = get_video_frames(video_path)
        if not frames:
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
        
        results["predictions"][filename] = response
        print(f"   ✅ Time: {t_end - t_start:.2f}s")
        print(f"   📝 {response[:200]}...")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, f"{model_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved to {args.output}/{model_name}_results.json")

if __name__ == "__main__":
    main()