import os
from huggingface_hub import HfApi, create_repo

# --- Configuration ---
ORG_NAME = "blind-assist"
MODEL_NAME = "internvl2-5-4b-walk-lora-v1-100" 
LOCAL_DIR = "work_dirs/internvl2_5_4b_walk_lora"
REPO_ID = f"{ORG_NAME}/{MODEL_NAME}"
BASE_MODEL = "OpenGVLab/InternVL2_5-4B"

api = HfApi()

# --- 1. Create Repo (if it doesn't exist) ---
print(f"Creating repo: {REPO_ID}...")
try:
    create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
except Exception as e:
    print(f"Note: {e}")

# --- 2. Generate Model Card (README.md) ---
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
This is a Fine-Tuned LoRA adapter for **InternVL2.5-4B**. 
It has been trained on the **WalkVLM** dataset to assist visually impaired individuals by detecting navigation hazards and providing scene descriptions.

## Intended Use
- **Task:** Visual Question Answering for Navigation
- **Domain:** Blind Assistance / Accessibility
- **Base Model:** [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})

## Training Details
- **Framework:** PyTorch & InternVL
- **Method:** LoRA (Low-Rank Adaptation)
- **Dataset:** WalkVLM (Custom subset)

## How to Use
```python
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

# Load Base Model
model = AutoModel.from_pretrained("{BASE_MODEL}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}", trust_remote_code=True)

# Load this Adapter
model = PeftModel.from_pretrained(model, "{REPO_ID}")

# Now you can use the model for inference!
```
"""

# --- 3. Write README.md to the local directory ---
readme_path = os.path.join(LOCAL_DIR, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)
print(f"✅ Model card saved to {readme_path}")

# --- 4. Upload Files ---
print(f"Uploading files from {LOCAL_DIR}...")
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    ignore_patterns=["checkpoint-*", "*.pth"]
)

print(f"✅ Upload Complete: https://huggingface.co/{REPO_ID}")