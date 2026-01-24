# CoVLA-Agent Implementation

**Paper**: [CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving](https://arxiv.org/pdf/2408.10845)  
**Datasets**: 
- [turing-motors/CoVLA-Dataset](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset) - Full (10,000 videos, ~12GB states)
- [turing-motors/CoVLA-Dataset-Mini](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset-Mini) - Mini (subset)

---

## Files

| File | Description |
|------|-------------|
| `covla_agent_paper.py` | **Main file** - Exact implementation of Section 4: Experiments |
| `./vla_data/` | Mini dataset directory |
| `./vla_data_full/` | Full dataset directory |

### Data Directory Structure (Mini)
```
./vla_data/
├── index.csv          # Video index
├── states/            # State JSONL files (ego_state, trajectory, etc.)
│   ├── video_id_1.jsonl
│   └── ...
├── captions/          # Caption files (plain_caption, rich_caption, etc.)
│   ├── video_id_1.jsonl
│   └── ...
└── images/            # Extracted images
    ├── video_folder_1/
    │   ├── 0000.png
    │   └── ...
    └── ...
```

### Data Directory Structure (Full)
```
./vla_data_full/
├── metadata.jsonl     # Video metadata (replaces index.csv)
├── states/            # Extracted from states.tar.gz (12GB)
│   ├── video_id_1.jsonl
│   └── ...
├── captions/          # Extracted from captions.tar.gz (69MB)
│   ├── video_id_1.jsonl
│   └── ...
├── front_car/         # Extracted from front_car.tar.gz (optional)
├── traffic_lights/    # Extracted from traffic_lights.tar.gz (optional)
└── videos/            # Video files (optional, for frame extraction)
    └── ...
```

---

## Paper Summary

CoVLA-Agent is a Vision-Language-Action model that:
1. **Predicts future trajectory** (10 waypoints over 3 seconds)
2. **Generates scene captions** (natural language descriptions)

### Architecture (from paper)
```
Image → CLIP ViT-L/14 → Projection ─┐
                                    ├→ Llama-2 7B → Trajectory MLP → 10 (x,y,z) points
Speed → Speed MLP ──────────────────┘            → Caption Generation
```

**Paper Models:**
- **Vision Encoder**: CLIP ViT-L/14 (768-dim features)
- **Language Model**: Llama-2 7B (we use Mistral 7B as open alternative)
- **Speed Embedding**: MLP that embeds ego vehicle speed

### Training (Section 4)
- **Loss**: `0.5 × Caption_CE + 0.5 × Trajectory_MSE`
- **Trajectory**: 10 points uniformly sampled from 60 (3-second horizon)
- **Frame sampling**: 2Hz
- **Split**: 70% train / 15% val / 15% test
- **Training samples**: 302,989

### Results (from paper)
| Setting | ADE | FDE |
|---------|-----|-----|
| Predicted captions | 0.955 | 2.239 |
| GT captions | 0.814 | 1.655 |

---

## Colab Tutorial

### Dataset Options

| Dataset | Videos | Frames | Size | Access |
|---------|--------|--------|------|--------|
| **Mini** (`CoVLA-Dataset-Mini`) | ~100 | ~60K | ~2GB | ✅ Public |
| **Full** (`CoVLA-Dataset`) | 10,000 | ~6M | ~15GB | 🔒 Approval Required |

- **Mini Dataset**: Use Cells 2-5 below (recommended for quick start)
- **Full Dataset**: Use Cells 2-F through 6-F (see "Full Dataset Loading" section below)

### Cell 1: Install Dependencies
```python
!pip install torch torchvision transformers peft huggingface_hub pandas matplotlib opencv-python
```

### Cell 2: Setup and Configuration (Mini Dataset)
```python
import json
import numpy as np
import os
import tarfile
import shutil
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files
import pandas as pd

REPO_ID = "turing-motors/CoVLA-Dataset-Mini"

# ============================================================
# CONFIGURATION
# ============================================================
NUM_VIDEOS = 40  # Set to desired number (more = better training)
                 # 1 = quick test (~600 frames)
                 # 10 = basic training (~6,000 frames)
                 # 40 = recommended (~24,000 frames)
                 # 100+ = best results (~60,000+ frames)

DATA_DIR = "./vla_data"  # Local directory to save data

# Create data directories
os.makedirs(f"{DATA_DIR}/states", exist_ok=True)
os.makedirs(f"{DATA_DIR}/captions", exist_ok=True)
os.makedirs(f"{DATA_DIR}/images", exist_ok=True)

# Download index (skip if exists)
local_index_path = f"{DATA_DIR}/index.csv"
if os.path.exists(local_index_path):
    print(f"✓ Using cached index: {local_index_path}")
else:
    index_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="index.csv")
    shutil.copy(index_path, local_index_path)
    print(f"✓ Downloaded index to: {local_index_path}")

index_df = pd.read_csv(local_index_path)
unique_video_ids = index_df['video_id'].unique().tolist()
print(f"Total videos available: {len(unique_video_ids)}")
print(f"Loading {min(NUM_VIDEOS, len(unique_video_ids))} videos from {DATA_DIR}/")
```

### Cell 3: Load States from Multiple Videos
```python
# Load states from multiple videos (skip download if cached)
all_states = []
video_ids_loaded = []
cached_count = 0
downloaded_count = 0

# Use unique video IDs from Cell 2
videos_to_load = unique_video_ids[:NUM_VIDEOS]

for i, video_id in enumerate(videos_to_load):
    local_states_path = f"{DATA_DIR}/states/{video_id}.jsonl"
    
    # Check if already downloaded
    if os.path.exists(local_states_path):
        with open(local_states_path, 'r') as f:
            video_states = [json.loads(line) for line in f]
        cached_count += 1
    else:
        try:
            states_path = hf_hub_download(
                repo_id=REPO_ID, 
                repo_type="dataset", 
                filename=f"states/{video_id}.jsonl"
            )
            shutil.copy(states_path, local_states_path)
            
            with open(local_states_path, 'r') as f:
                video_states = [json.loads(line) for line in f]
            downloaded_count += 1
        except Exception as e:
            print(f"  Skipped video {video_id}: {e}")
            continue
    
    all_states.extend(video_states)
    video_ids_loaded.append(video_id)
    
    if (i + 1) % 10 == 0:
        print(f"  Loaded {i + 1}/{len(videos_to_load)} videos ({len(all_states)} frames)")

states = all_states
print(f"\n✓ Loaded {len(states)} total frames from {len(video_ids_loaded)} videos")
print(f"  (cached: {cached_count}, downloaded: {downloaded_count})")
print(f"State keys: {states[0].keys()}")
```

### Cell 4: Load Captions from Multiple Videos
```python
# Load captions from multiple videos (skip download if cached)
all_captions = []
cached_count = 0
downloaded_count = 0

for video_id in video_ids_loaded:
    local_captions_path = f"{DATA_DIR}/captions/{video_id}.jsonl"
    
    # Check if already downloaded
    if os.path.exists(local_captions_path):
        with open(local_captions_path, "r") as f:
            raw = f.read()
        cached_count += 1
    else:
        try:
            captions_path = hf_hub_download(
                repo_id=REPO_ID, 
                repo_type="dataset", 
                filename=f"captions/{video_id}.jsonl"
            )
            shutil.copy(captions_path, local_captions_path)
            
            with open(local_captions_path, "r") as f:
                raw = f.read()
            downloaded_count += 1
        except Exception as e:
            continue
    
    # Parse concatenated JSON objects
    parts = raw.split('}{')
    for j, part in enumerate(parts):
        if j == 0: s = part + '}'
        elif j == len(parts)-1: s = '{' + part
        else: s = '{' + part + '}'
        try:
            all_captions.append(json.loads(s))
        except:
            pass

captions_data = all_captions
print(f"✓ Loaded {len(captions_data)} captions")
print(f"  (cached: {cached_count}, downloaded: {downloaded_count})")
if captions_data:
    print(f"Sample: {captions_data[0]['plain_caption'][:100]}...")
```

### Cell 5: Extract Images for Loaded Videos
```python
# Get list of all image archives from repo
files = list_repo_files(REPO_ID, repo_type="dataset")
all_image_archives = [f for f in files if f.startswith("images/") and f.endswith(".tar.gz")]

# Extract images to local directory
images_dir = f"{DATA_DIR}/images"
os.makedirs(images_dir, exist_ok=True)

# Check for existing images first
existing_images = sorted([
    os.path.join(root, f)
    for root, dirs, files_list in os.walk(images_dir)
    for f in files_list if f.endswith(('.jpg', '.png', '.jpeg'))
])

if existing_images:
    print(f"✓ Using {len(existing_images)} cached images from {images_dir}/")
    print("  (delete folder to re-download)")
    image_files = existing_images
else:
    # Find archives matching loaded video IDs
    archives_to_extract = []
    for video_id in video_ids_loaded:
        matching = [a for a in all_image_archives if video_id in a]
        archives_to_extract.extend(matching)
    
    print(f"Extracting images for {len(video_ids_loaded)} videos ({len(archives_to_extract)} archives)")

    for i, archive_name in enumerate(archives_to_extract):
        try:
            archive_path = hf_hub_download(
                repo_id=REPO_ID, 
                repo_type="dataset", 
                filename=archive_name
            )
            
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(images_dir)
            
            if (i + 1) % 10 == 0:
                print(f"  Extracted {i + 1}/{len(archives_to_extract)} archives")
                
        except Exception as e:
            print(f"  Skipped {archive_name}: {e}")

    # Collect all image files
    image_files = sorted([
        os.path.join(root, f)
        for root, dirs, files_list in os.walk(images_dir)
        for f in files_list if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    print(f"\n✓ Extracted {len(image_files)} images to {images_dir}/")
```

---

## Full Dataset Loading (Alternative)

If you want to use the **full CoVLA-Dataset** (10,000 videos, ~80 hours of driving) instead of the Mini dataset:

> ⚠️ **Note**: Full dataset requires ~15GB download and access approval from [Hugging Face](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset)

### Prerequisites for Full Dataset

1. **Request Access**: Go to [turing-motors/CoVLA-Dataset](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset) and click "Access repository" to accept the license terms.

2. **Hugging Face Login** (in Colab):
```python
from huggingface_hub import login
login()  # Enter your HF token when prompted
# Or use: !huggingface-cli login
```

### Cell 2-F: Setup for Full Dataset
```python
import json
import numpy as np
import os
import tarfile
import shutil
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
import pandas as pd

# ============================================================
# FULL DATASET CONFIGURATION
# ============================================================
REPO_ID = "turing-motors/CoVLA-Dataset"  # Full dataset
DATA_DIR = "./vla_data_full"  # Different directory from mini

# Create data directories
os.makedirs(f"{DATA_DIR}/states", exist_ok=True)
os.makedirs(f"{DATA_DIR}/captions", exist_ok=True)

# Number of videos to use (full dataset has 10,000)
NUM_VIDEOS = 100  # Set higher for better training
                  # 100 = ~60,000 frames
                  # 1000 = ~600,000 frames  
                  # 10000 = full dataset (~6M frames)

print(f"Data directory: {DATA_DIR}")
print(f"Target videos: {NUM_VIDEOS}")
```

### Cell 3-F: Download and Extract States (12GB tar.gz)
```python
# Download metadata.jsonl (replaces index.csv in mini dataset)
metadata_path = f"{DATA_DIR}/metadata.jsonl"
if os.path.exists(metadata_path):
    print(f"✓ Using cached metadata: {metadata_path}")
else:
    print("Downloading metadata.jsonl...")
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID, 
        repo_type="dataset", 
        filename="metadata.jsonl"
    )
    shutil.copy(downloaded_path, metadata_path)
    print(f"✓ Downloaded metadata to: {metadata_path}")

# Parse metadata to get video IDs
with open(metadata_path, 'r') as f:
    metadata = [json.loads(line) for line in f]
video_ids = [m['video_id'] for m in metadata]
print(f"Total videos in full dataset: {len(video_ids)}")

# Download and extract states.tar.gz (12GB - this takes a while!)
states_dir = f"{DATA_DIR}/states"
states_marker = f"{states_dir}/.extracted"

if os.path.exists(states_marker):
    print(f"✓ States already extracted to {states_dir}/")
else:
    print("Downloading states.tar.gz (~12GB, this may take 10-30 minutes)...")
    states_archive = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename="states.tar.gz"
    )
    
    print("Extracting states (this also takes a while)...")
    with tarfile.open(states_archive, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    
    # Create marker file
    with open(states_marker, 'w') as f:
        f.write("extracted")
    print(f"✓ Extracted states to {states_dir}/")
```

### Cell 4-F: Download and Extract Captions (69MB tar.gz)
```python
# Download and extract captions.tar.gz
captions_dir = f"{DATA_DIR}/captions"
captions_marker = f"{captions_dir}/.extracted"

if os.path.exists(captions_marker):
    print(f"✓ Captions already extracted to {captions_dir}/")
else:
    print("Downloading captions.tar.gz (~69MB)...")
    captions_archive = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename="captions.tar.gz"
    )
    
    print("Extracting captions...")
    with tarfile.open(captions_archive, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    
    with open(captions_marker, 'w') as f:
        f.write("extracted")
    print(f"✓ Extracted captions to {captions_dir}/")
```

### Cell 5-F: Load States and Captions from Full Dataset
```python
# Helper to handle both flat and frame-indexed JSONL formats
def parse_jsonl_line(line_data):
    first_key = next(iter(line_data.keys()), None)
    if first_key and first_key.isdigit() and 'ego_state' not in line_data:
        return line_data[first_key]
    return line_data

# Load states from multiple videos
all_states = []
all_captions = []
video_ids_loaded = []

videos_to_load = video_ids[:NUM_VIDEOS]

for i, video_id in enumerate(videos_to_load):
    # Load states
    states_path = f"{DATA_DIR}/states/{video_id}.jsonl"
    if not os.path.exists(states_path):
        continue
    
    with open(states_path, 'r') as f:
        video_states = [parse_jsonl_line(json.loads(line)) for line in f if line.strip()]
    
    if not video_states:
        continue
    
    # Load captions
    video_captions = []
    captions_path = f"{DATA_DIR}/captions/{video_id}.jsonl"
    if os.path.exists(captions_path):
        with open(captions_path, 'r') as f:
            video_captions = [parse_jsonl_line(json.loads(line)) for line in f if line.strip()]
    
    # Pad captions to match states
    if not video_captions:
        video_captions = [{}]
    while len(video_captions) < len(video_states):
        video_captions.append(video_captions[-1])
    
    all_states.extend(video_states)
    all_captions.extend(video_captions[:len(video_states)])
    video_ids_loaded.append(video_id)
    
    if (i + 1) % 50 == 0:
        print(f"  Loaded {i + 1}/{len(videos_to_load)} videos ({len(all_states)} frames)")

states = all_states
captions_data = all_captions
print(f"\n✓ Loaded {len(states)} frames from {len(video_ids_loaded)} videos")
print(f"✓ Loaded {len(captions_data)} captions")

# Sample at EXTRACT_FPS (must match Cell 7-F)
EXTRACT_FPS = 2
FRAME_INTERVAL = 20 // EXTRACT_FPS  # 10 for 2Hz
states = states[::FRAME_INTERVAL]
captions_data = captions_data[::FRAME_INTERVAL]
print(f"✓ Sampled at {EXTRACT_FPS}Hz: {len(states)} states, {len(captions_data)} captions")
```

### Cell 6-F: Download Videos
```python
# NOTE: The full dataset stores videos, not pre-extracted images.
# This cell downloads videos. Run Cell 7-F to extract frames.

from huggingface_hub import list_repo_files

# Get list of available videos
video_files = [f for f in list_repo_files(REPO_ID, repo_type="dataset") 
               if f.startswith("videos/") and f.endswith(".mp4")]
print(f"Found {len(video_files)} videos in dataset")

# Create videos directory
videos_dir = f"{DATA_DIR}/videos"
os.makedirs(videos_dir, exist_ok=True)

# Configuration: how many videos to download
NUM_VIDEOS_TO_DOWNLOAD = 100  # Adjust as needed (each video ~30s, ~50-100MB)

# Filter to videos we have states for
videos_to_download = [vid for vid in video_ids_loaded[:NUM_VIDEOS_TO_DOWNLOAD]
                      if f"videos/{vid}.mp4" in video_files]

print(f"Will download {len(videos_to_download)} videos")

# Download videos (skip if already exists)
downloaded_count = 0
cached_count = 0

for i, video_id in enumerate(videos_to_download):
    local_video = f"{videos_dir}/{video_id}.mp4"
    
    if os.path.exists(local_video):
        cached_count += 1
        continue
    
    try:
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"videos/{video_id}.mp4"
        )
        shutil.copy(downloaded, local_video)
        downloaded_count += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Downloaded {i + 1}/{len(videos_to_download)} videos")
            
    except Exception as e:
        print(f"  Could not download {video_id}: {e}")

print(f"\n✓ Videos ready: {downloaded_count} downloaded, {cached_count} cached")
print(f"  Location: {videos_dir}/")
```

### Cell 7-F: Extract Frames from Videos
```python
# Extract frames from downloaded videos using OpenCV (no ffmpeg needed!)
# Paper (Section 4.1): "We sample frames at a frequency of 2Hz"
# - 20Hz extraction: 600 frames/video (full resolution for flexibility)
# - 2Hz extraction: 60 frames/video (paper's training rate, 10x less storage)

import cv2

frames_dir = f"{DATA_DIR}/frames"
os.makedirs(frames_dir, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================
EXTRACT_FPS = 2         # Paper uses 2Hz for training (set to 20 for full rate)
DELETE_VIDEOS = False   # Keep videos (2Hz extraction is already space-efficient)

def extract_frames_from_video(video_id, fps=EXTRACT_FPS, delete_after=DELETE_VIDEOS):
    """
    Extract frames from a single video using OpenCV.
    Frames are named with ORIGINAL indices to match states JSONL.
    
    Args:
        video_id: Video identifier
        fps: Frames per second to extract (2 = paper rate, 20 = full)
        delete_after: Delete MP4 after successful extraction
    
    Returns:
        (output_dir, success): Path to extracted frames and success flag
    
    Frame naming (for 30s video at 20Hz original):
        - 20Hz extraction: 0000.png, 0001.png, ... 0599.png (600 frames)
        - 2Hz extraction:  0000.png, 0010.png, ... 0590.png (60 frames)
    """
    video_path = f"{DATA_DIR}/videos/{video_id}.mp4"
    output_dir = f"{frames_dir}/{video_id}"
    
    # Skip if already extracted
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        return output_dir, True
    
    # Check video exists
    if not os.path.exists(video_path):
        return None, False
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open video: {video_id}")
        return None, False
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        original_fps = 20  # Default assumption
    
    # Calculate frame interval
    frame_interval = int(round(original_fps / fps))  # 10 for 2Hz from 20Hz
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every Nth frame with original index as filename
        if frame_idx % frame_interval == 0:
            filename = f"{output_dir}/{frame_idx:04d}.png"
            cv2.imwrite(filename, frame)
            saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    
    # Delete video after successful extraction
    if delete_after and saved_count > 0 and os.path.exists(video_path):
        os.remove(video_path)
    
    return output_dir, saved_count > 0

# Only process videos that have states loaded (from Cell 5-F)
videos_to_extract = [vid for vid in video_ids_loaded 
                     if os.path.exists(f"{DATA_DIR}/videos/{vid}.mp4")]

print(f"Videos with states loaded: {len(video_ids_loaded)}")
print(f"Videos with MP4 available: {len(videos_to_extract)}")
print(f"Settings: {EXTRACT_FPS}Hz extraction, delete_after={DELETE_VIDEOS}")

# Extract frames
extracted_count = 0
cached_count = 0

for i, video_id in enumerate(videos_to_extract):
    output_dir = f"{frames_dir}/{video_id}"
    
    # Check if already extracted
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        cached_count += 1
        continue
    
    output_dir, success = extract_frames_from_video(video_id)
    if success:
        extracted_count += 1
        
    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(videos_to_extract)} videos")

print(f"\n✓ Frames ready: {extracted_count} extracted, {cached_count} cached")

# Collect image files only from loaded videos
image_files = []
for video_id in video_ids_loaded:
    video_frames_dir = f"{frames_dir}/{video_id}"
    if os.path.exists(video_frames_dir):
        for f in sorted(os.listdir(video_frames_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(video_frames_dir, f))

print(f"✓ Total frames available: {len(image_files)}")
print(f"  From {len(video_ids_loaded)} videos with states")
print(f"  Location: {frames_dir}/")
```

### Frame-to-State Mapping (Simplified!)

Frames are now named with **original indices** that directly match states JSONL:

| Extraction | Frame Names | State Index |
|------------|-------------|-------------|
| **20Hz** | `0000.png`, `0001.png`, ... `0599.png` | `state_idx = frame_num` |
| **2Hz** | `0000.png`, `0010.png`, ... `0590.png` | `state_idx = frame_num` |

```python
# Simple mapping - frame name IS the state index!
def get_state_index(frame_filename):
    """Frame name directly matches state index."""
    return int(frame_filename.split('.')[0])  # "0010.png" -> 10

# Example usage
frame = "0050.png"
state_idx = get_state_index(frame)  # -> 50
state = states[state_idx]  # Direct lookup!
```

### Cell 8-F: Visualize Raw Data (before dataset creation)
```python
from covla_agent_paper import visualize_sample

# Visualize raw loaded data (states, captions_data, image_files)
visualize_sample(0, image_files, states, captions_data)
visualize_sample(30, image_files, states, captions_data)

# Check multiple samples
for idx in [0, 10, 30, 50]:
    if idx < len(image_files):
        visualize_sample(idx, image_files, states, captions_data)
```

### Cell 8-F (alt): Visualize Dataset Samples (after dataset creation)
```python
from covla_agent_paper import visualize_dataset_sample

# Visualize samples from train/val dataset
visualize_dataset_sample(val_dataset, 0)
visualize_dataset_sample(val_dataset, 10)
visualize_dataset_sample(train_dataset, 50)

# Check multiple validation samples
for idx in [0, 5, 10, 20]:
    if idx < len(val_dataset):
        print(f"\n{'='*50}")
        visualize_dataset_sample(val_dataset, idx)
```

### Alternative: Batch Download with snapshot_download
For downloading many videos at once, use `snapshot_download`:
```python
from huggingface_hub import snapshot_download

# Download all videos (WARNING: This is ~100GB+ for full dataset!)
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns="videos/*",
    local_dir=DATA_DIR,
    max_workers=4  # Parallel downloads
)
```

### Alternative: Use Pre-extracted Images from Mini Dataset
Since the full dataset requires video frame extraction, you can also:
1. Use the Mini dataset for images (pre-extracted)
2. Use the Full dataset for states/captions (more annotations)

---

### Cell 6a: Setup for Paper Model (7B LLM) - RECOMMENDED
```python
# Install dependencies
!pip install bitsandbytes accelerate

# Note: The default is now Mistral-7B which is OPEN (no approval needed!)
# If you want Llama-2 instead, you need to:
# 1. Go to: https://huggingface.co/meta-llama/Llama-2-7b-hf
# 2. Click "Access repository" and accept Meta's license
# 3. Wait for approval (usually instant)
# 4. Run: !huggingface-cli login
```

### Cell 6b: Create Model and Train
```python
from covla_agent_paper import *

# ============================================================
# Option A: Paper Model (RECOMMENDED for better captions)
# Requires: ~24GB VRAM (A100) or ~16GB with 8-bit quantization
# ============================================================
config = CoVLAConfig(
    device="cuda",
    use_paper_model=True,   # CLIP ViT-L + Llama-2 7B
    use_speed_embedding=True,
    batch_size=4,           # Reduce if OOM (default: 8)
)

# ============================================================
# Option B: Lightweight Model (for free Colab T4 GPU)
# Works with ~8GB VRAM but caption quality is lower
# ============================================================
# config = CoVLAConfig(
#     device="cuda",
#     use_paper_model=False,  # CLIP ViT-B + TinyLlama 1.1B
#     use_speed_embedding=True,
#     batch_size=4,           # Reduce if OOM (default: 8)
# )

# Create datasets
train_dataset = CoVLADatasetPaper(states, captions_data, image_files, config, split="train")
val_dataset = CoVLADatasetPaper(states, captions_data, image_files, config, split="val")

# Create model
model = CoVLAAgentPaper(config)

# Train (auto-saves to covla_model.pt)
trainer = CoVLATrainerPaper(model, config)
history = trainer.train(train_dataset, val_dataset, num_epochs=10)

# Manual save (optional)
# trainer.save_checkpoint("my_model.pt")
```

### Cell 6c: Load Saved Model (skip training)
```python
from covla_agent_paper import load_model

# Load previously trained model
model = load_model("covla_model.pt")

# Or with custom path
# model = load_model("/path/to/my_model.pt", device="cuda")
```

### Model Comparison

| Setting | Vision | Language | VRAM | Caption Quality | Approval |
|---------|--------|----------|------|-----------------|----------|
| `use_paper_model=True` | CLIP ViT-L/14 | Mistral 7B | ~24GB | ⭐⭐⭐ Best | ✅ None |
| `use_paper_model=False` | CLIP ViT-B/32 | TinyLlama 1.1B | ~8GB | ⭐ Basic | ✅ None |

**Alternative 7B models** (change in config if needed):
- `mistralai/Mistral-7B-Instruct-v0.2` - Open, no approval (DEFAULT)
- `meta-llama/Llama-2-7b-hf` - Requires Meta approval
- `NousResearch/Llama-2-7b-hf` - Community mirror, no approval

### Memory Tips
- **Colab Pro+ with A100**: Use `use_paper_model=True` directly
- **Colab Pro with V100**: May need 8-bit quantization
- **Free Colab T4**: Use `use_paper_model=False`

### Cell 7: Evaluate and Visualize
```python
# Plot training curves
plot_training_curves(history)

# Single prediction (generates caption → uses it for trajectory)
sample = val_dataset[0]
result = model.predict(
    sample['image'], 
    speed=sample['speed'],  # REQUIRED
    caption_mode="pred",  # Generate caption, use for trajectory (default)
    # caption_mode="gt", caption=sample.get('caption'),  # Or use GT caption (better ADE)
)
print(f"Trajectory shape: {result['trajectory'].shape}")  # (10, 3)
print(f"Caption used: {result['caption'][:100]}...")
```

### Cell 8: Visualize Predictions
```python
from covla_agent_paper import visualize

# Simple API - just pass model, dataset, and index!
visualize(model, val_dataset, idx=0)

# With GT caption (oracle mode - better ADE)
visualize(model, val_dataset, idx=0, caption_mode="gt")
```

### Cell 9: Visualize Multiple Samples
```python
for idx in [0, 10, 20]:
    if idx < len(val_dataset):
        print(f"\n{'='*50}\nSample {idx}\n{'='*50}")
        visualize(model, val_dataset, idx)
```

### Cell 10: Generate Video
```python
from covla_agent_paper import generate_video

generate_video(
    model, 
    val_dataset, 
    output_path="demo.mp4", 
    num_frames=30,          # Number of frames
    fps=2,                  # Output FPS
    caption_mode="pred",    # "pred" or "gt"
    show_gt=True,           # Show GT trajectory (green)
)
```

**Output video shows:**
- 🟢 Green: Ground truth trajectory
- 🔴 Red: Predicted trajectory  
- Top-left: Speed, ADE/FDE metrics
- Bottom: Generated caption

---

## Metrics

### ADE (Average Displacement Error)
Mean Euclidean distance between predicted and ground truth points over all time steps.

```python
ADE = mean(√((pred_x - gt_x)² + (pred_y - gt_y)² + (pred_z - gt_z)²))
```

### FDE (Final Displacement Error)
Euclidean distance between predicted and ground truth final points.

```python
FDE = √((pred_x[-1] - gt_x[-1])² + (pred_y[-1] - gt_y[-1])² + (pred_z[-1] - gt_z[-1])²)
```

---

## Data Format

### States (per frame)
```json
{
  "ego_state": {
    "vEgo": 6.37,           // Speed (m/s)
    "steeringAngleDeg": -2.6, // Steering angle
    "gas": 0.17,            // Gas pedal
    "brake": 0.0            // Brake pedal
  },
  "trajectory": [[x,y,z], ...],  // 60 waypoints (3 sec @ 20Hz)
  "extrinsic_matrix": [...],     // Camera extrinsic (3x4)
  "intrinsic_matrix": [...]      // Camera intrinsic (3x3)
}
```

### Captions (per frame)
```json
{
  "plain_caption": "The ego vehicle is moving straight...",
  "rich_caption": "The ego vehicle is moving straight... The driver should...",
  "weather": "rainy",
  "road": "wide road",
  "is_tunnel": false,
  "is_highway": true,
  "has_pedestrian": false
}
```

---

## Requirements

```
torch>=2.0
torchvision
transformers>=4.30
peft
huggingface_hub
pandas
matplotlib
numpy
Pillow
```

---

## Citation

```bibtex
@article{covla2024,
  title={CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving},
  author={Arai, Hidehisa and Miwa, Keita and Sasaki, Kento and others},
  journal={arXiv preprint arXiv:2408.10845},
  year={2024}
}
```

