# # Cell: Test your TRAINED model (it's a full fine-tuned model, not LoRA)

# import json
# import time
# import torch
# import glob
# import os
# import cv2
# import numpy as np
# from typing import List
# from PIL import Image as PILImage
# from transformers import AutoModel, AutoTokenizer
# import torchvision.transforms as T
# from torchvision.transforms.functional import InterpolationMode

# # Your trained model path (local)
# TRAINED_MODEL_PATH = "work_dirs/internvl2_5_4b_walk_lora"
# BASE_MODEL = "OpenGVLab/InternVL2_5-4B"

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

# def load_model(model_path, is_local=True):
#     """Load model from local path or HuggingFace."""
#     print(f"🔄 Loading model from: {model_path}")
    
#     model = AutoModel.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#         local_files_only=is_local
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path, 
#         trust_remote_code=True,
#         local_files_only=is_local
#     )
    
#     model.eval()
#     print("✅ Model loaded!")
#     return model, tokenizer

# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", type=str, default="./my_videos")
#     parser.add_argument("--output", type=str, default="./inference_results")
#     parser.add_argument("--base", action="store_true", help="Use base model")
#     parser.add_argument("--model_path", type=str, default=None, help="Custom model path")
#     args = parser.parse_args()

#     # Determine which model to load
#     if args.base:
#         model_path = BASE_MODEL
#         is_local = False
#         model_name = "base"
#     elif args.model_path:
#         model_path = args.model_path
#         is_local = os.path.exists(model_path)
#         model_name = "custom"
#     else:
#         model_path = TRAINED_MODEL_PATH
#         is_local = True
#         model_name = "trained"

#     print(f"📁 Input folder: {args.input}")
    
#     video_paths = sorted(glob.glob(os.path.join(args.input, "*.mp4")) + 
#                          glob.glob(os.path.join(args.input, "*.avi")))
    
#     if not video_paths:
#         print(f"❌ No videos found")
#         return

#     print(f"🚀 Found {len(video_paths)} videos")
#     model, tokenizer = load_model(model_path, is_local)
    
#     prompt = "You are an assistive navigation system for a visually impaired user. Analyze this scene and identify all immediate, high-risk obstructions. State each obstruction's location using the 12-hour clock face. Generate a single, actionable safety alert."
    
#     results = {"model": model_name, "model_path": model_path, "predictions": {}}

#     for idx, video_path in enumerate(video_paths[:5]):  # Test first 5
#         filename = os.path.basename(video_path)
#         print(f"\n📦 Processing {idx+1}: {filename}")
        
#         frames = get_video_frames(video_path)
#         if not frames:
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

#     os.makedirs(args.output, exist_ok=True)
#     with open(os.path.join(args.output, f"{model_name}_results.json"), "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\n✅ Saved to {args.output}/{model_name}_results.json")

# if __name__ == "__main__":
#     main()



















# # Cell: Test your TRAINED model (LoRA fine-tuned with InternVL training script)

# import json
# import time
# import torch
# import glob
# import os
# import cv2
# import numpy as np
# from typing import List
# from PIL import Image as PILImage
# from transformers import AutoModel, AutoTokenizer
# from peft import LoraConfig, get_peft_model
# from safetensors.torch import load_file
# import torchvision.transforms as T
# from torchvision.transforms.functional import InterpolationMode

# # Your trained model path (local)
# TRAINED_MODEL_PATH = "work_dirs/internvl2_5_4b_walk_lora"
# BASE_MODEL = "OpenGVLab/InternVL2_5-4B"

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

# def load_lora_weights_manually(model, checkpoint_path):
#     """Load LoRA weights from checkpoint that was saved with InternVL training script."""
#     print(f"🔄 Loading checkpoint weights from: {checkpoint_path}")
    
#     # Find safetensor files
#     safetensor_files = sorted(glob.glob(os.path.join(checkpoint_path, "model*.safetensors")))
    
#     if not safetensor_files:
#         raise FileNotFoundError(f"No safetensor files found in {checkpoint_path}")
    
#     # Load all weights
#     all_weights = {}
#     for sf_file in safetensor_files:
#         print(f"   Loading: {os.path.basename(sf_file)}")
#         weights = load_file(sf_file)
#         all_weights.update(weights)
    
#     # Get model state dict
#     model_state = model.state_dict()
    
#     # Map checkpoint keys to model keys
#     # Checkpoint format: language_model.base_model.model.model.layers.X.self_attn.q_proj.base_layer.weight
#     # Model format: language_model.model.layers.X.self_attn.q_proj.weight
    
#     loaded_count = 0
#     skipped_count = 0
    
#     new_state_dict = {}
#     for ckpt_key, ckpt_value in all_weights.items():
#         # Skip LoRA weights (lora_A, lora_B) - we need to handle them separately
#         if 'lora_A' in ckpt_key or 'lora_B' in ckpt_key:
#             skipped_count += 1
#             continue
            
#         # Convert checkpoint key to model key
#         # Remove 'base_model.model.' from the path
#         model_key = ckpt_key.replace('base_model.model.', '')
#         # Also handle 'base_layer.' which wraps the original weight
#         model_key = model_key.replace('.base_layer', '')
        
#         if model_key in model_state:
#             if model_state[model_key].shape == ckpt_value.shape:
#                 new_state_dict[model_key] = ckpt_value
#                 loaded_count += 1
#             else:
#                 print(f"   ⚠️ Shape mismatch for {model_key}: model={model_state[model_key].shape}, ckpt={ckpt_value.shape}")
#         else:
#             # Try without language_model prefix variations
#             pass
    
#     # Load the mapped weights
#     missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
#     print(f"   ✅ Loaded {loaded_count} weight tensors")
#     print(f"   ⚠️ Skipped {skipped_count} LoRA-specific tensors")
#     print(f"   📝 Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
#     return model

# def load_model_with_merged_lora(checkpoint_path, base_model_name):
#     """
#     Load the checkpoint which contains base weights + LoRA weights merged in a specific format.
#     The InternVL training saves the full model state with LoRA layers intact.
#     """
#     print(f"🔄 Loading base model: {base_model_name}")
    
#     # Load base model
#     model = AutoModel.from_pretrained(
#         base_model_name,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )
    
#     tokenizer = AutoTokenizer.from_pretrained(
#         base_model_name, 
#         trust_remote_code=True,
#     )
    
#     # Load and apply checkpoint weights
#     model = load_lora_weights_manually(model, checkpoint_path)
    
#     model.eval()
#     print("✅ Model loaded with fine-tuned weights!")
#     return model, tokenizer

# def load_base_model_only(base_model_name):
#     """Load just the base model without any fine-tuning (for comparison)."""
#     print(f"🔄 Loading base model: {base_model_name}")
    
#     model = AutoModel.from_pretrained(
#         base_model_name,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )
    
#     tokenizer = AutoTokenizer.from_pretrained(
#         base_model_name, 
#         trust_remote_code=True,
#     )
    
#     model.eval()
#     print("✅ Base model loaded!")
#     return model, tokenizer

# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", type=str, default="./my_videos")
#     parser.add_argument("--output", type=str, default="./inference_results")
#     parser.add_argument("--base", action="store_true", help="Use base model only (no fine-tuning)")
#     parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
#     args = parser.parse_args()

#     # Determine which model to load
#     if args.base:
#         model, tokenizer = load_base_model_only(BASE_MODEL)
#         model_name = "base"
#     else:
#         checkpoint_path = args.checkpoint if args.checkpoint else TRAINED_MODEL_PATH
#         model, tokenizer = load_model_with_merged_lora(checkpoint_path, BASE_MODEL)
#         model_name = "finetuned"

#     print(f"📁 Input folder: {args.input}")
    
#     video_paths = sorted(glob.glob(os.path.join(args.input, "*.mp4")) + 
#                          glob.glob(os.path.join(args.input, "*.avi")))
    
#     if not video_paths:
#         print(f"❌ No videos found")
#         return

#     print(f"🚀 Found {len(video_paths)} videos")
    
#     prompt = "You are an assistive navigation system for a visually impaired user. Analyze this scene and identify all immediate, high-risk obstructions. State each obstruction's location using the 12-hour clock face. Generate a single, actionable safety alert."
    
#     results = {"model": model_name, "predictions": {}}

#     for idx, video_path in enumerate(video_paths[:5]):  # Test first 5
#         filename = os.path.basename(video_path)
#         print(f"\n📦 Processing {idx+1}: {filename}")
        
#         frames = get_video_frames(video_path)
#         if not frames:
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

#     os.makedirs(args.output, exist_ok=True)
#     with open(os.path.join(args.output, f"{model_name}_results.json"), "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\n✅ Saved to {args.output}/{model_name}_results.json")

# if __name__ == "__main__":
#     main()

























# import json
# import time
# import torch
# import glob
# import os
# import cv2
# import numpy as np
# from typing import List
# from PIL import Image as PILImage
# from transformers import AutoModel, AutoTokenizer
# from peft import LoraConfig, get_peft_model
# from safetensors.torch import load_file
# import torchvision.transforms as T
# from torchvision.transforms.functional import InterpolationMode

# # Your trained model path (local)
# TRAINED_MODEL_PATH = "work_dirs/internvl2_5_4b_walk_lora"
# BASE_MODEL = "OpenGVLab/InternVL2_5-4B"

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

# def load_lora_weights_manually(model, checkpoint_path):
#     """Load LoRA weights from checkpoint that was saved with InternVL training script."""
#     print(f"🔄 Loading checkpoint weights from: {checkpoint_path}")
    
#     # Find safetensor files
#     safetensor_files = sorted(glob.glob(os.path.join(checkpoint_path, "model*.safetensors")))
    
#     if not safetensor_files:
#         raise FileNotFoundError(f"No safetensor files found in {checkpoint_path}")
    
#     # Load all weights
#     all_weights = {}
#     for sf_file in safetensor_files:
#         print(f"   Loading: {os.path.basename(sf_file)}")
#         weights = load_file(sf_file)
#         all_weights.update(weights)
    
#     # Get model state dict
#     model_state = model.state_dict()
    
#     # Map checkpoint keys to model keys
#     # Checkpoint format: language_model.base_model.model.model.layers.X.self_attn.q_proj.base_layer.weight
#     # Model format: language_model.model.layers.X.self_attn.q_proj.weight
    
#     loaded_count = 0
#     skipped_count = 0
    
#     new_state_dict = {}
#     for ckpt_key, ckpt_value in all_weights.items():
#         # Skip LoRA weights (lora_A, lora_B) - we need to handle them separately
#         if 'lora_A' in ckpt_key or 'lora_B' in ckpt_key:
#             skipped_count += 1
#             continue
            
#         # Convert checkpoint key to model key
#         # Remove 'base_model.model.' from the path
#         model_key = ckpt_key.replace('base_model.model.', '')
#         # Also handle 'base_layer.' which wraps the original weight
#         model_key = model_key.replace('.base_layer', '')
        
#         if model_key in model_state:
#             if model_state[model_key].shape == ckpt_value.shape:
#                 new_state_dict[model_key] = ckpt_value
#                 loaded_count += 1
#             else:
#                 print(f"   ⚠️ Shape mismatch for {model_key}: model={model_state[model_key].shape}, ckpt={ckpt_value.shape}")
#         else:
#             # Try without language_model prefix variations
#             pass
    
#     # Load the mapped weights
#     missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
#     print(f"   ✅ Loaded {loaded_count} weight tensors")
#     print(f"   ⚠️ Skipped {skipped_count} LoRA-specific tensors")
#     print(f"   📝 Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
#     return model

# def load_model_with_merged_lora(checkpoint_path, base_model_name):
#     """
#     Load the checkpoint which contains base weights + LoRA weights merged in a specific format.
#     The InternVL training saves the full model state with LoRA layers intact.
#     """
#     print(f"🔄 Loading base model: {base_model_name}")
    
#     # Load base model
#     model = AutoModel.from_pretrained(
#         base_model_name,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )
    
#     tokenizer = AutoTokenizer.from_pretrained(
#         base_model_name, 
#         trust_remote_code=True,
#     )
    
#     # Load and apply checkpoint weights
#     model = load_lora_weights_manually(model, checkpoint_path)
    
#     model.eval()
#     print("✅ Model loaded with fine-tuned weights!")
#     return model, tokenizer

# def load_base_model_only(base_model_name):
#     """Load just the base model without any fine-tuning (for comparison)."""
#     print(f"🔄 Loading base model: {base_model_name}")
    
#     model = AutoModel.from_pretrained(
#         base_model_name,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )
    
#     tokenizer = AutoTokenizer.from_pretrained(
#         base_model_name, 
#         trust_remote_code=True,
#     )
    
#     model.eval()
#     print("✅ Base model loaded!")
#     return model, tokenizer

# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", type=str, default="./my_videos")
#     parser.add_argument("--output", type=str, default="./inference_results")
#     parser.add_argument("--base", action="store_true", help="Use base model only (no fine-tuning)")
#     parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
#     parser.add_argument("--max_videos", type=int, default=None, help="Max videos to process (default: all)")
#     args = parser.parse_args()

#     # Determine which model to load
#     if args.base:
#         model, tokenizer = load_base_model_only(BASE_MODEL)
#         model_name = "base"
#     else:
#         checkpoint_path = args.checkpoint if args.checkpoint else TRAINED_MODEL_PATH
#         model, tokenizer = load_model_with_merged_lora(checkpoint_path, BASE_MODEL)
#         model_name = "finetuned"

#     print(f"📁 Input folder: {args.input}")
    
#     video_paths = sorted(glob.glob(os.path.join(args.input, "*.mp4")) + 
#                          glob.glob(os.path.join(args.input, "*.avi")))
    
#     if not video_paths:
#         print(f"❌ No videos found")
#         return

#     # Apply max_videos limit if specified
#     if args.max_videos:
#         video_paths = video_paths[:args.max_videos]
    
#     print(f"🚀 Found {len(video_paths)} videos to process")
    
    
#     prompt = "You are an assistive navigation system for a visually impaired user. Analyze this scene and identify all immediate, high-risk obstructions. State each obstruction's location using the 12-hour clock face. Generate a single, actionable safety alert."
    
#     results = {"model": model_name, "predictions": {}}

#     for idx, video_path in enumerate(video_paths):  # Process all videos
#         filename = os.path.basename(video_path)
#         print(f"\n📦 Processing {idx+1}/{len(video_paths)}: {filename}")
        
#         frames = get_video_frames(video_path)
#         if not frames:
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

#     os.makedirs(args.output, exist_ok=True)
#     with open(os.path.join(args.output, f"{model_name}_results.json"), "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\n✅ Saved to {args.output}/{model_name}_results.json")

# if __name__ == "__main__":
#     main()


































# Test your TRAINED model with MERGED LoRA weights

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
from safetensors.torch import load_file
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

def load_and_merge_lora_weights(model, checkpoint_path):
    """
    Load checkpoint and MERGE LoRA weights into base weights.
    Formula: W_merged = W_base + (lora_B @ lora_A) * scaling
    """
    print(f"🔄 Loading checkpoint weights from: {checkpoint_path}")
    
    safetensor_files = sorted(glob.glob(os.path.join(checkpoint_path, "model*.safetensors")))
    
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in {checkpoint_path}")
    
    # Load all weights from checkpoint
    all_weights = {}
    for sf_file in safetensor_files:
        print(f"   Loading: {os.path.basename(sf_file)}")
        weights = load_file(sf_file)
        all_weights.update(weights)
    
    print(f"   📊 Total checkpoint tensors: {len(all_weights)}")
    
    # Separate weights by type
    base_weights = {}      # Original layer weights (wrapped in base_layer)
    lora_a_weights = {}    # LoRA A matrices
    lora_b_weights = {}    # LoRA B matrices
    other_weights = {}     # Vision encoder, embeddings, etc.
    
    for key, value in all_weights.items():
        if '.lora_A.' in key:
            lora_a_weights[key] = value
        elif '.lora_B.' in key:
            lora_b_weights[key] = value
        elif '.base_layer.' in key:
            base_weights[key] = value
        else:
            other_weights[key] = value
    
    print(f"   📊 Base layer weights: {len(base_weights)}")
    print(f"   📊 LoRA A weights: {len(lora_a_weights)}")
    print(f"   📊 LoRA B weights: {len(lora_b_weights)}")
    print(f"   📊 Other weights: {len(other_weights)}")
    
    # Get model state dict
    model_state = model.state_dict()
    
    # Merge LoRA into base weights
    merged_count = 0
    scaling = 1.0  # LoRA scaling factor (typically lora_alpha / lora_r)
    
    new_state_dict = {}
    
    for base_key, base_value in base_weights.items():
        # Find matching LoRA weights
        # base_key: language_model.base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
        # lora_a:   language_model.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
        # lora_b:   language_model.base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
        
        lora_a_key = base_key.replace('.base_layer.', '.lora_A.default.')
        lora_b_key = base_key.replace('.base_layer.', '.lora_B.default.')
        
        # Convert to model key format
        # FROM: language_model.base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
        # TO:   language_model.model.layers.0.self_attn.q_proj.weight
        model_key = base_key.replace('base_model.model.', '').replace('.base_layer', '')
        
        if lora_a_key in lora_a_weights and lora_b_key in lora_b_weights:
            # Merge LoRA: W' = W + B @ A * scaling
            lora_a = lora_a_weights[lora_a_key].float()
            lora_b = lora_b_weights[lora_b_key].float()
            base = base_value.float()
            
            # LoRA merge: output = base + (lora_b @ lora_a) * scaling
            delta = torch.matmul(lora_b, lora_a) * scaling
            merged = base + delta
            
            new_state_dict[model_key] = merged.to(base_value.dtype)
            merged_count += 1
        else:
            # No LoRA for this weight, use base directly
            new_state_dict[model_key] = base_value
    
    # Add other weights (vision encoder, embeddings, layernorm, etc.)
    for key, value in other_weights.items():
        model_key = key.replace('base_model.model.', '')
        new_state_dict[model_key] = value
    
    # Load merged weights into model
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
    """Load base model and apply MERGED LoRA weights."""
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
    
    # Load and MERGE LoRA weights (not skip!)
    model = load_and_merge_lora_weights(model, checkpoint_path)
    
    model.eval()
    print("✅ Model loaded with MERGED LoRA fine-tuned weights!")
    return model, tokenizer

def load_base_model_only(base_model_name):
    """Load just the base model without any fine-tuning (for comparison)."""
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




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./my_videos")
    parser.add_argument("--output", type=str, default="./inference_results")
    parser.add_argument("--base", action="store_true", help="Use base model only (no fine-tuning)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    parser.add_argument("--max_videos", type=int, default=None, help="Max videos to process")
    args = parser.parse_args()

    # Determine which model to load
    if args.base:
        model, tokenizer = load_base_model_only(BASE_MODEL)
        model_name = "base"
    else:
        checkpoint_path = args.checkpoint if args.checkpoint else TRAINED_MODEL_PATH
        model, tokenizer = load_model_with_merged_lora(checkpoint_path, BASE_MODEL)
        model_name = "finetuned"

    print(f"\n📁 Input folder: {args.input}")
    
    video_paths = sorted(glob.glob(os.path.join(args.input, "*.mp4")) + 
                         glob.glob(os.path.join(args.input, "*.avi")))
    
    if not video_paths:
        print(f"❌ No videos found")
        return

    if args.max_videos:
        video_paths = video_paths[:args.max_videos]
    
    print(f"🚀 Processing {len(video_paths)} videos")
    
    prompt = "You are an assistive navigation system for a visually impaired user. Analyze this scene and identify all immediate, high-risk obstructions. State each obstruction's location using the 12-hour clock face. Generate a single, actionable safety alert."
    
    # Store results for Excel
    results_list = []
    results = {"model": model_name, "predictions": {}}

    for idx, video_path in enumerate(video_paths):
        filename = os.path.basename(video_path)
        print(f"\n📦 Processing {idx+1}/{len(video_paths)}: {filename}")
        
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
                response_escaped = r["response"].replace('"', '""')
                f.write(f'"{r["filename"]}","{response_escaped}",{r["inference_time_sec"]},"{r["model"]}","{r["prompt"]}"\n')
        print(f"✅ CSV saved to {csv_file}")

if __name__ == "__main__":
    main()

# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", type=str, default="./my_videos")
#     parser.add_argument("--output", type=str, default="./inference_results")
#     parser.add_argument("--base", action="store_true", help="Use base model only (no fine-tuning)")
#     parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
#     parser.add_argument("--max_videos", type=int, default=None, help="Max videos to process")
#     args = parser.parse_args()

#     # Determine which model to load
#     if args.base:
#         model, tokenizer = load_base_model_only(BASE_MODEL)
#         model_name = "base"
#     else:
#         checkpoint_path = args.checkpoint if args.checkpoint else TRAINED_MODEL_PATH
#         model, tokenizer = load_model_with_merged_lora(checkpoint_path, BASE_MODEL)
#         model_name = "finetuned"

#     print(f"\n📁 Input folder: {args.input}")
    
#     video_paths = sorted(glob.glob(os.path.join(args.input, "*.mp4")) + 
#                          glob.glob(os.path.join(args.input, "*.avi")))
    
#     if not video_paths:
#         print(f"❌ No videos found")
#         return

#     if args.max_videos:
#         video_paths = video_paths[:args.max_videos]
    
#     print(f"🚀 Processing {len(video_paths)} videos")
    
#     prompt = "You are an assistive navigation system for a visually impaired user. Analyze this scene and identify all immediate, high-risk obstructions. State each obstruction's location using the 12-hour clock face. Generate a single, actionable safety alert."
    
#     results = {"model": model_name, "predictions": {}}

#     for idx, video_path in enumerate(video_paths):
#         filename = os.path.basename(video_path)
#         print(f"\n📦 Processing {idx+1}/{len(video_paths)}: {filename}")
        
#         frames = get_video_frames(video_path)
#         if not frames:
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

#     os.makedirs(args.output, exist_ok=True)
#     output_file = os.path.join(args.output, f"{model_name}_results.json")
#     with open(output_file, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\n✅ Saved to {output_file}")

# if __name__ == "__main__":
    main()