import os
import glob
import time
import cv2
import numpy as np
from PIL import Image as PILImage

# Import the high-speed inference engine
from lmdeploy import pipeline, TurbomindEngineConfig

# --- Configuration ---
MODEL_REPO = "blind-assist/internvl3-2b-walk-awq"
INPUT_FOLDER = "/content/my_videos"

def get_middle_frame(video_path: str) -> PILImage.Image:
    """Extracts the middle frame of a video and returns it as a PIL Image."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_idx = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(frame)
    return None

def main():
    print(f"🚀 Loading AWQ Model via LMDeploy Engine from {MODEL_REPO}...")
    
    # Configure the engine for 4-bit AWQ format
    backend_config = TurbomindEngineConfig(model_format='awq')
    
    # The pipeline handles all the messy preprocessing and tokenization for us!
    pipe = pipeline(MODEL_REPO, backend_config=backend_config)
    print("✅ AWQ Model loaded into GPU memory!")

    prompt = ("Given the visual input from the user’s forward perspective, identify the closest immediate "
              "obstacle that poses the highest collision risk (especially within approximately 2 meters), "
              "and generate exactly one short sentence guiding a visually impaired user by describing its "
              "location using clock directions relative to the user (12 o’clock is straight ahead), "
              "including relevant details such as size, material, or distance, and giving one clear action "
              "to avoid it, prioritizing immediate safety and ignoring less urgent or distant objects, "
              "with no extra explanation.")

    # Find videos and images
    print(f"\n📁 Scanning folder: {INPUT_FOLDER}")
    file_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.*")))
    file_paths = [p for p in file_paths if p.lower().endswith(('.mp4', '.avi', '.jpg', '.png', '.jpeg'))]

    if not file_paths:
        print(f"❌ No videos or images found in {INPUT_FOLDER}")
        return

    print(f"🚀 Processing {len(file_paths)} files\n" + "-"*50)

    for idx, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        print(f"📦 [{idx+1}/{len(file_paths)}] Processing: {filename}")
        
        # Load the image or extract the middle frame
        if file_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            image = PILImage.open(file_path).convert("RGB")
        else:
            image = get_middle_frame(file_path)
            
        if image is None:
            print("   ⚠️ Could not extract frame.")
            continue

        # --- THE INFERENCE STEP ---
        start_time = time.time()
        
        # LMDeploy takes a tuple of (prompt, PIL_Image)
        response = pipe((prompt, image))
        
        elapsed = time.time() - start_time
        
        print(f"   ⏱️ Inference Time: {elapsed:.2f} seconds")
        print(f"   📝 Alert: {response.text}\n")

if __name__ == "__main__":
    main()