import os
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import argparse

# --- MONKEY PATCH 1: Tied Weights Fix ---
try:
    import transformers.quantizers.base
    original_get_keys = transformers.quantizers.base.get_keys_to_not_convert
    def patched_get_keys(model, *args, **kwargs):
        if not hasattr(model, 'all_tied_weights_keys'):
            model.all_tied_weights_keys = {}
        return original_get_keys(model, *args, **kwargs)
    transformers.quantizers.base.get_keys_to_not_convert = patched_get_keys

    import transformers.modeling_utils
    original_get_total_byte_count = transformers.modeling_utils.get_total_byte_count
    def patched_get_total_byte_count(model, *args, **kwargs):
        if not hasattr(model, 'all_tied_weights_keys'):
            model.all_tied_weights_keys = {}
        return original_get_total_byte_count(model, *args, **kwargs)
    transformers.modeling_utils.get_total_byte_count = patched_get_total_byte_count
except ImportError:
    pass

# --- Configuration ---
BASE_MODEL = "OpenGVLab/InternVL3-2B" 
LORA_REPO = "blind-assist/internvl3-2b-walk-lora-Epoch3-8500-v2"

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into InternVL3 Base Model")
    parser.add_argument("--output_dir", type=str, default="./internvl3-2b-merged", help="Where to save the merged model")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the merged model to Hugging Face")
    parser.add_argument("--hub_repo_id", type=str, default="", help="Your HF repo ID (e.g., 'your-username/internvl3-merged')")
    args = parser.parse_args()

    print(f"🚀 Starting Merge Process...")
    print(f"📦 Base Model: {BASE_MODEL}")
    print(f"🧩 LoRA Adapter: {LORA_REPO}")

    # --- MONKEY PATCH 2: InternVL Meta-Device Bug Fix ---
    original_linspace = torch.linspace
    def safe_linspace(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return original_linspace(*args, **kwargs)
    torch.linspace = safe_linspace
    # ----------------------------------------------------

    try:
        print("\n🔄 Step 1: Loading Base Model in 16-bit (bfloat16)...")
        # Notice we are NOT using quantization_config here. Pure 16-bit for merging.
        model = AutoModel.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            device_map={"": 0}, # Load directly to Colab's GPU
            low_cpu_mem_usage=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, 
            trust_remote_code=True
        )
    finally:
        torch.linspace = original_linspace # Restore PyTorch behavior

    print("\n🔄 Step 2: Attaching LoRA Adapter...")
    model = PeftModel.from_pretrained(model, LORA_REPO)

    print("\n🔥 Step 3: Fusing Weights (Merge and Unload)...")
    # This takes the adapter weights and permanently adds them to the base model weights
    merged_model = model.merge_and_unload()
    print("✅ Weights successfully merged!")

    print(f"\n💾 Step 4: Saving Merged Model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the model and tokenizer locally
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Local save complete!")

    # Optional: Push to Hugging Face directly
    if args.push_to_hub and args.hub_repo_id:
        print(f"\n☁️ Step 5: Pushing to Hugging Face Hub ({args.hub_repo_id})...")
        merged_model.push_to_hub(args.hub_repo_id, safe_serialization=True)
        tokenizer.push_to_hub(args.hub_repo_id)
        print("✅ Successfully pushed to Hugging Face!")

if __name__ == "__main__":
    main()