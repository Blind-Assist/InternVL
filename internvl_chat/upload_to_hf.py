# import os
# from huggingface_hub import HfApi, create_repo

# # --- Configuration ---
# ORG_NAME = "blind-assist"
# MODEL_NAME = "internvl2-5-4b-walk-lora-v1-100" 
# LOCAL_DIR = "work_dirs/internvl2_5_4b_walk_lora"
# REPO_ID = f"{ORG_NAME}/{MODEL_NAME}"
# BASE_MODEL = "OpenGVLab/InternVL2_5-4B"

# api = HfApi()

# # --- 1. Create Repo (if it doesn't exist) ---
# print(f"Creating repo: {REPO_ID}...")
# try:
#     create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
# except Exception as e:
#     print(f"Note: {e}")

# # --- 2. Generate Model Card (README.md) ---
# print("📝 Generating Model Card...")

# readme_content = f"""---
# library_name: peft
# base_model: {BASE_MODEL}
# tags:
# - internvl
# - vision
# - image-text-to-text
# - lora
# - blind-assist
# - walk-vlm
# datasets:
# - blind-assist/walk-train
# ---

# # {MODEL_NAME}

# ## Model Description
# This is a Fine-Tuned LoRA adapter for **InternVL2.5-4B**. 
# It has been trained on the **WalkVLM** dataset to assist visually impaired individuals by detecting navigation hazards and providing scene descriptions.

# ## Intended Use
# - **Task:** Visual Question Answering for Navigation
# - **Domain:** Blind Assistance / Accessibility
# - **Base Model:** [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})

# ## Training Details
# - **Framework:** PyTorch & InternVL
# - **Method:** LoRA (Low-Rank Adaptation)
# - **Dataset:** WalkVLM (Custom subset)

# ## How to Use
# ```python
# from peft import PeftModel
# from transformers import AutoModel, AutoTokenizer

# # Load Base Model
# model = AutoModel.from_pretrained("{BASE_MODEL}", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}", trust_remote_code=True)

# # Load this Adapter
# model = PeftModel.from_pretrained(model, "{REPO_ID}")

# # Now you can use the model for inference!
# ```
# """

# # --- 3. Write README.md to the local directory ---
# readme_path = os.path.join(LOCAL_DIR, "README.md")
# with open(readme_path, "w", encoding="utf-8") as f:
#     f.write(readme_content)
# print(f"✅ Model card saved to {readme_path}")

# # --- 4. Upload Files ---
# print(f"Uploading files from {LOCAL_DIR}...")
# api.upload_folder(
#     folder_path=LOCAL_DIR,
#     repo_id=REPO_ID,
#     repo_type="model",
#     ignore_patterns=["checkpoint-*", "*.pth"]
# )

# print(f"✅ Upload Complete: https://huggingface.co/{REPO_ID}")

















import os
import json
import torch
import glob
from huggingface_hub import HfApi, create_repo
from safetensors.torch import load_file, save_file

# --- Configuration ---
ORG_NAME = "blind-assist"
MODEL_NAME = "internvl2-5-4b-walk-lora-v2-100" 
LOCAL_DIR = "work_dirs/internvl2_5_4b_walk_lora"
PEFT_OUTPUT_DIR = "work_dirs/internvl2_5_4b_walk_lora_peft"  # Converted PEFT format
REPO_ID = f"{ORG_NAME}/{MODEL_NAME}"
BASE_MODEL = "OpenGVLab/InternVL2_5-4B"

api = HfApi()

# --- 1. Convert to Standard PEFT Format ---
def convert_to_peft_format():
    """Extract LoRA weights and save in standard PEFT format."""
    print("🔄 Converting checkpoint to standard PEFT format...")
    
    os.makedirs(PEFT_OUTPUT_DIR, exist_ok=True)
    
    # Load checkpoint
    safetensor_files = sorted(glob.glob(os.path.join(LOCAL_DIR, "model*.safetensors")))
    all_weights = {}
    for sf_file in safetensor_files:
        print(f"   Loading: {os.path.basename(sf_file)}")
        weights = load_file(sf_file)
        all_weights.update(weights)
    
    # Extract LoRA weights
    lora_weights = {}
    target_modules = set()
    lora_r = None
    
    for key, value in all_weights.items():
        if '.lora_A.' in key or '.lora_B.' in key:
            # Convert key format for PEFT
            new_key = key
            new_key = new_key.replace('language_model.base_model.model.', 'base_model.model.language_model.')
            new_key = new_key.replace('.lora_A.default.', '.lora_A.')
            new_key = new_key.replace('.lora_B.default.', '.lora_B.')
            
            lora_weights[new_key] = value
            
            # Extract target module names
            if '.lora_A.' in key:
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == 'lora_A':
                        target_modules.add(parts[i-1])
                        break
                if lora_r is None:
                    lora_r = value.shape[0]
    
    print(f"   📊 Extracted {len(lora_weights)} LoRA tensors")
    print(f"   📊 Target modules: {target_modules}")
    print(f"   📊 LoRA rank: {lora_r}")
    
    # Create adapter_config.json
    adapter_config = {
        "auto_mapping": None,
        "base_model_name_or_path": BASE_MODEL,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": lora_r,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_r,
        "revision": None,
        "target_modules": list(target_modules),
        "task_type": "CAUSAL_LM"
    }
    
    # Save adapter_config.json
    with open(os.path.join(PEFT_OUTPUT_DIR, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
    
    # Save LoRA weights
    save_file(lora_weights, os.path.join(PEFT_OUTPUT_DIR, "adapter_model.safetensors"))
    
    print(f"   ✅ Saved PEFT adapter to {PEFT_OUTPUT_DIR}")
    return PEFT_OUTPUT_DIR

# --- 2. Create Repo ---
print(f"Creating repo: {REPO_ID}...")
try:
    create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
except Exception as e:
    print(f"Note: {e}")

# --- 3. Convert to PEFT format ---
peft_dir = convert_to_peft_format()

# --- 4. Generate Model Card ---
print("📝 Generating Model Card...")

readme_content = f"""---
library_name: peft
base_model: {BASE_MODEL}
tags:
- internvl
- vision
- image-text-to-text
- lora
- blind-assist
- walk-vlm
datasets:
- blind-assist/walk-train
---

# {MODEL_NAME}

## Model Description
This is a **LoRA adapter** for **InternVL2.5-4B**, fine-tuned on the **WalkVLM** dataset to assist visually impaired individuals with navigation hazard detection.

## How to Use

### Method 1: Using PEFT (Recommended)
```python
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

# Load Base Model
base_model = AutoModel.from_pretrained(
    "{BASE_MODEL}",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}", trust_remote_code=True)

# Load LoRA Adapter
model = PeftModel.from_pretrained(base_model, "{REPO_ID}")

# Merge for faster inference (optional)
model = model.merge_and_unload()

# Use for inference
response = model.chat(
    tokenizer=tokenizer,
    pixel_values=pixel_values,  # Your preprocessed image
    question="Describe any obstacles in this scene.",
    generation_config=dict(max_new_tokens=256)
)
```

### Method 2: Manual LoRA Merge
If PEFT doesn't work due to model architecture, use manual merging:
```python
# See our inference script at:
# https://github.com/Blind-Assist/InternVL/blob/walkvlm/internvl_chat/test_finetuned_model.py
```

## Training Details
- **Base Model:** [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})
- **Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank:** 16
- **Dataset:** [blind-assist/walk-train](https://huggingface.co/datasets/blind-assist/walk-train)
- **Task:** Navigation hazard detection for visually impaired users

## Files
- `adapter_config.json` - PEFT LoRA configuration
- `adapter_model.safetensors` - LoRA weights only (~50MB)

## License
Same as base model ({BASE_MODEL})
"""

readme_path = os.path.join(peft_dir, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)
print(f"✅ Model card saved")

# --- 5. Upload PEFT Format ---
print(f"📤 Uploading PEFT adapter from {peft_dir}...")
api.upload_folder(
    folder_path=peft_dir,
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"\n✅ Upload Complete!")
print(f"🔗 https://huggingface.co/{REPO_ID}")
print(f"\n📝 Users can now load with:")
print(f'   model = PeftModel.from_pretrained(base_model, "{REPO_ID}")')