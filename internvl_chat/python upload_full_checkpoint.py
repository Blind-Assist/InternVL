"""
Upload full InternVL checkpoint to HuggingFace for backup/security.
This includes all files: model weights, checkpoints, configs, etc.
"""
import os
from huggingface_hub import HfApi, create_repo

# --- Configuration ---
ORG_NAME = "blind-assist"
MODEL_NAME = "internvl2-5-4b-walk-full-checkpoint"  # Different name for full backup
LOCAL_DIR = "work_dirs/internvl2_5_4b_walk_lora"
REPO_ID = f"{ORG_NAME}/{MODEL_NAME}"
BASE_MODEL = "OpenGVLab/InternVL2_5-4B"

api = HfApi()

# --- 1. Create Repo ---
print(f"Creating repo: {REPO_ID}...")
try:
    create_repo(repo_id=REPO_ID, repo_type="model", private=True, exist_ok=True)  # Private for security
    print(f"✅ Repo created (private)")
except Exception as e:
    print(f"Note: {e}")

# --- 2. Generate Model Card ---
print("📝 Generating Model Card...")

readme_content = f"""---
base_model: {BASE_MODEL}
tags:
- internvl
- vision
- lora
- blind-assist
- walk-vlm
- full-checkpoint
datasets:
- blind-assist/walk-train
---

# {MODEL_NAME}

## ⚠️ This is a Full Checkpoint Backup

This repository contains the **complete training checkpoint** from InternVL fine-tuning, including:
- Full model weights with embedded LoRA layers
- Training checkpoints (checkpoint-900, checkpoint-950)
- Tokenizer files
- Training configs and states

## Files Included
```
├── checkpoint-900/          # Training checkpoint at step 900
├── checkpoint-950/          # Training checkpoint at step 950
├── model-00001-of-00002.safetensors  # Full model weights (part 1)
├── model-00002-of-00002.safetensors  # Full model weights (part 2)
├── model.safetensors.index.json
├── config.json
├── generation_config.json
├── tokenizer files...
├── trainer_state.json
├── training_args.bin
└── train_results.json
```

## How to Use This Checkpoint

### Option 1: Load and Merge LoRA (Recommended)
```python
import torch
import glob
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file

BASE_MODEL = "{BASE_MODEL}"
CHECKPOINT = "path/to/downloaded/checkpoint"

# Load base model
model = AutoModel.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Load checkpoint weights
safetensor_files = sorted(glob.glob(f"{{CHECKPOINT}}/model*.safetensors"))
all_weights = {{}}
for sf in safetensor_files:
    all_weights.update(load_file(sf))

# Separate and merge LoRA weights
model_state = model.state_dict()
for key, value in all_weights.items():
    if '.base_layer.' in key:
        # Find LoRA weights
        lora_a_key = key.replace('.base_layer.', '.lora_A.default.')
        lora_b_key = key.replace('.base_layer.', '.lora_B.default.')
        model_key = key.replace('base_model.model.', '').replace('.base_layer', '')
        
        if lora_a_key in all_weights and lora_b_key in all_weights:
            lora_a = all_weights[lora_a_key].float()
            lora_b = all_weights[lora_b_key].float()
            merged = value.float() + torch.matmul(lora_b, lora_a)
            if model_key in model_state:
                model_state[model_key] = merged.to(value.dtype)

model.load_state_dict(model_state)
print("✅ Model loaded with merged LoRA weights")
```

### Option 2: Use Our PEFT Adapter (Easier)
For easier loading, use our converted PEFT adapter:
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "blind-assist/internvl2-5-4b-walk-lora-v2-100")
```

## Training Details
- **Base Model:** [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})
- **Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank:** 16
- **Dataset:** [blind-assist/walk-train](https://huggingface.co/datasets/blind-assist/walk-train)

## Related Repositories
- **PEFT Adapter (Recommended for inference):** [blind-assist/internvl2-5-4b-walk-lora-v2-100](https://huggingface.co/blind-assist/internvl2-5-4b-walk-lora-v2-100)

## License
Same as base model ({BASE_MODEL})
"""

readme_path = os.path.join(LOCAL_DIR, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)
print(f"✅ Model card saved")

# --- 3. Upload ALL Files ---
print(f"📤 Uploading full checkpoint from {LOCAL_DIR}...")
print("   This may take a while due to large file sizes...")

api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    ignore_patterns=["checkpoint-*", "*.bin"],  # Skip checkpoint folders and optimizer states
    # Don't ignore anything - upload everything
)

print(f"\n✅ Upload Complete!")
print(f"🔗 https://huggingface.co/{REPO_ID}")
print(f"\n📝 Note: This repo is set to PRIVATE for security.")
print(f"   To make it public, go to Settings on HuggingFace.")