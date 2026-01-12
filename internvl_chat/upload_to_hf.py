import os
from huggingface_hub import HfApi, create_repo

# --- Configuration ---
ORG_NAME = "blind-assist"
MODEL_NAME = "internvl2-5-4b-walk-lora-v1" # You can change v1 to v2, etc.
LOCAL_DIR = "work_dirs/internvl2_5_4b_walk_lora"
REPO_ID = f"{ORG_NAME}/{MODEL_NAME}"

api = HfApi()

# 1. Create Repo (if it doesn't exist)
print(f"Creating repo: {REPO_ID}...")
try:
    create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
except Exception as e:
    print(f"Note: {e}")

# 2. Upload Files
print(f"Uploading files from {LOCAL_DIR}...")
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    ignore_patterns=["checkpoint-*", "*.pth"] # Skip massive checkpoints if you only want the final adapter
)

print(f"✅ Upload Complete: https://huggingface.co/{REPO_ID}")