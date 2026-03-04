# CoVLA-Agent Implementation

**Paper**: [CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving](https://arxiv.org/pdf/2408.10845)  
**Datasets**: 
- [turing-motors/CoVLA-Dataset](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset) - Full (10,000 videos, ~12GB states)
- [turing-motors/CoVLA-Dataset-Mini](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset-Mini) - Mini (subset)

## Quick Start

**Prerequisites:** Load `states`, `captions_data`, `image_files`, and `lead_car_data` (see Full pipeline below). CoT uses **lead car only** (R2 first, then caption); traffic lights are skipped for now.

```python
from covla_agent_paper import *

config = CoVLAConfig(device="cuda", model_size="paper")
# lead_car_data required; traffic_light_data optional (not used in R2 currently)
train_dataset = CoVLADatasetPaper(states, captions_data, image_files, config, split="train",
    lead_car_data=lead_car_data, traffic_light_data=None)
val_dataset = CoVLADatasetPaper(states, captions_data, image_files, config, split="val",
    lead_car_data=lead_car_data, traffic_light_data=None)

model = CoVLAAgentPaper(config)
trainer = CoVLATrainerPaper(model, config)
history = trainer.train(train_dataset, val_dataset, num_epochs=4)
generate_eval_images(model, val_dataset, "eval", num_frames=100)
```

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
├── front_car/         # Extracted from front_car.tar.gz (required for CoT R2 — lead car)
├── traffic_lights/    # Extracted from traffic_lights.tar.gz (optional; not used in R2 yet)
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
Image → CLIP ViT-L/14 → Projection ──────────┐
                                             ├→ Mistral 7B → Trajectory MLP → 10 (x,y,z) points
Ego State → Linear Embedding ────────────────┘            → Caption Generation
[vEgo, aEgo, steering]
```

**Models:**
- **Vision Encoder**: CLIP ViT-L/14 (frozen, 768-dim features)
- **Language Model**: Mistral 7B with LoRA fine-tuning
- **Ego State**: Linear embedding of [speed, acceleration, steering angle]

### Training (Section 4)
- **Loss**: `0.5 × Caption_CE + 0.5 × Trajectory_MSE + R2_weight × R2_CE` (R2 = extra CE on lead-car tokens)
- **Trajectory**: 10 points uniformly sampled from 60 (3-second horizon)
- **Frame sampling**: 2Hz
- **Split**: 80% train / 20% val (configurable)
- **Learning rate**: 2e-5 with cosine decay to 1/3 of original
- **Data augmentation**: Image color jitter, trajectory noise, ego state noise
- **Quantization**: 8-bit or 4-bit QLoRA to reduce VRAM and increase batch size

### Results
| Setting | ADE | FDE | Notes |
|---------|-----|-----|-------|
| Paper (pred captions) | 0.955 | 2.239 | Baseline |
| Paper (GT captions) | 0.814 | 1.655 | Oracle |
| **Our best (6k clips)** | **0.724** | **1.774** | With augmentation |

---

## Configuration Options

### CoVLAConfig Parameters

```python
config = CoVLAConfig(
    # Model selection
    model_size="paper",          # "light" (TinyLlama), "paper" (Mistral 7B), "mixtral" (8x7B MoE)
    
    # Quantization (QLoRA) — reduces LLM VRAM, enables larger batch size
    quantize="8bit",             # "none" (FP16), "8bit", or "4bit"
    # Mistral 7B: FP16 ~14GB → 8bit ~7GB → 4bit ~3.5GB
    # Estimated batch size on 24GB GPU: FP16 ~4-5 → 8bit ~10-12 → 4bit ~16-20
    
    # Training
    learning_rate=2e-5,          # Initial LR (cosine decay to 1/3)
    batch_size=8,                # Increase with quantize="8bit" or "4bit"
    num_epochs=10,
    
    # Data split
    train_ratio=0.80,            # 80% train
    val_ratio=0.20,              # 20% val
    test_ratio=0.0,              # No test set
    
    # Data augmentation (training only)
    augment_trajectory=True,     # Gaussian noise on GT trajectory
    augment_ego_state=True,      # Multiplicative noise on ego state
    augment_image=True,          # Color jitter (brightness, contrast, saturation)
    trajectory_noise_std=0.05,   # 5cm std
    ego_state_noise_std=0.02,    # 2% relative noise
    
    # Loss weights
    caption_weight=0.5,
    trajectory_weight=0.5,
    smoothing_weight=0.1,        # Penalize trajectory acceleration
    caption_r2_weight=1.0,       # Extra R2 loss weight (CE on lead-car tokens)
    
    # Ego state
    use_extended_ego_state=True, # Use [vEgo, aEgo, steering] instead of just speed
    
    # LoRA fine-tuning
    use_lora=True,
    lora_rank=16,
)
```

### Data Augmentation

| Type | Augmentation | Default | Effect |
|------|--------------|---------|--------|
| **Image** | ColorJitter | On | Robustness to lighting |
| **Trajectory** | Gaussian noise (σ=5cm) | On | Prevents overfitting |
| **Ego state** | Multiplicative (2%) | On | Sensor noise robustness |

Augmentation only applies to training set; validation stays clean.

### Quantization (QLoRA)

Quantize the base LLM to reduce VRAM and increase batch size. LoRA adapters remain in FP16.

| Mistral 7B | FP16 (`"none"`) | 8-bit (`"8bit"`) | 4-bit (`"4bit"`) |
|------------|-----------------|-------------------|-------------------|
| LLM memory | ~14GB | ~7GB | ~3.5GB |
| Batch size (24GB GPU) | 4-5 | 10-12 | 16-20 |
| Accuracy impact | baseline | minimal | slight degradation |

```python
# 8-bit (recommended — best accuracy/memory tradeoff)
config = CoVLAConfig(model_size="paper", quantize="8bit", batch_size=10)

# 4-bit (maximum memory savings)
config = CoVLAConfig(model_size="paper", quantize="4bit", batch_size=16)
```

Existing checkpoints are compatible — they only contain LoRA + heads, not the base model weights. Requires `bitsandbytes` (`pip install bitsandbytes`).

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

### Cell 4-G: Download and Extract front_car and traffic_lights (optional, for CoT R2)
```python
# front_car.tar.gz (~132MB) — lead vehicle per frame
front_car_dir = f"{DATA_DIR}/front_car"
front_car_marker = f"{front_car_dir}/.extracted"
if os.path.exists(front_car_marker):
    print(f"✓ front_car already extracted to {front_car_dir}/")
else:
    print("Downloading front_car.tar.gz (~132MB)...")
    front_car_archive = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename="front_car.tar.gz"
    )
    print("Extracting front_car...")
    with tarfile.open(front_car_archive, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    with open(front_car_marker, 'w') as f:
        f.write("extracted")
    print(f"✓ Extracted front_car to {front_car_dir}/")

# traffic_lights.tar.gz (~65.5MB) — traffic light states per frame
traffic_lights_dir = f"{DATA_DIR}/traffic_lights"
traffic_lights_marker = f"{traffic_lights_dir}/.extracted"
if os.path.exists(traffic_lights_marker):
    print(f"✓ traffic_lights already extracted to {traffic_lights_dir}/")
else:
    print("Downloading traffic_lights.tar.gz (~65.5MB)...")
    traffic_lights_archive = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename="traffic_lights.tar.gz"
    )
    print("Extracting traffic_lights...")
    with tarfile.open(traffic_lights_archive, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    with open(traffic_lights_marker, 'w') as f:
        f.write("extracted")
    print(f"✓ Extracted traffic_lights to {traffic_lights_dir}/")
```

### Cell 5-F: Load States, Captions, and Front Car (CoT)
```python
# Load states, captions, and front_car (required for CoT R2 — lead car only).
# traffic_lights can be loaded optionally for a future experiment; not used in R2 yet.
all_states = []
all_captions = []
all_lead_car = []
all_traffic_lights = []
video_ids_loaded = []

front_car_dir = f"{DATA_DIR}/front_car"
traffic_lights_dir = f"{DATA_DIR}/traffic_lights"
has_front_car = os.path.isdir(front_car_dir)
has_traffic_lights = os.path.isdir(traffic_lights_dir)

videos_to_load = video_ids[:NUM_VIDEOS]

for i, video_id in enumerate(videos_to_load):
    # Load states
    states_path = f"{DATA_DIR}/states/{video_id}.jsonl"
    if not os.path.exists(states_path):
        continue
    
    video_states = []
    with open(states_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            assert isinstance(row, dict), f"states expected dict, got {type(row)}"
            first_key = next(iter(row.keys()), None)
            assert first_key and first_key.isdigit(), f"states expected digit key, got {first_key!r}"
            video_states.append(row[first_key])
    
    if not video_states:
        continue
    
    # Load captions (JSONL: one line = {"0": {plain_caption, ...}} — dict with digit key)
    captions_path = f"{DATA_DIR}/captions/{video_id}.jsonl"
    if os.path.exists(captions_path):
        with open(captions_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                assert isinstance(row, dict), f"captions expected dict, got {type(row)}"
                first_key = next(iter(row.keys()), None)
                assert first_key and first_key.isdigit(), f"captions expected digit key, got {first_key!r}"
                all_captions.append(row[first_key])
    
    # Pad captions to match this video's state count (align by frame)
    n_states = len(video_states)
    while len(all_captions) < len(all_states) + n_states:
        all_captions.append(all_captions[-1] if all_captions else {})
    
    # Load front_car (one line = one frame; GT: {"0": {has_lead, ...}} — dict with digit key)
    video_lead_car = []
    if has_front_car:
        fc_path = f"{front_car_dir}/{video_id}.jsonl"
        if os.path.exists(fc_path):
            with open(fc_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    assert isinstance(row, dict), f"front_car expected dict, got {type(row)}"
                    first_key = next(iter(row.keys()), None)
                    assert first_key and first_key.isdigit(), f"front_car expected digit key, got {first_key!r}"
                    video_lead_car.append(row[first_key])
    while len(video_lead_car) < n_states:
        video_lead_car.append({})
    video_lead_car = video_lead_car[:n_states]  # trim if CoT file had more lines than state file
    
    # Load traffic_lights (one line = one frame; GT: {"0": [...] or None} — dict with digit key)
    video_tlights = []
    if has_traffic_lights:
        tl_path = f"{traffic_lights_dir}/{video_id}.jsonl"
        if os.path.exists(tl_path):
            with open(tl_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    assert isinstance(row, dict), f"traffic_lights expected dict, got {type(row)}"
                    first_key = next(iter(row.keys()), None)
                    assert first_key and first_key.isdigit(), f"traffic_lights expected digit key, got {first_key!r}"
                    val = row[first_key]
                    video_tlights.append(val if val is not None else [])
    while len(video_tlights) < n_states:
        video_tlights.append([])
    video_tlights = video_tlights[:n_states]  # trim if CoT file had more lines than state file
    
    all_states.extend(video_states)
    all_lead_car.extend(video_lead_car[:n_states])
    all_traffic_lights.extend(video_tlights[:n_states])
    video_ids_loaded.append(video_id)
    
    if (i + 1) % 50 == 0:
        print(f"  Loaded {i + 1}/{len(videos_to_load)} videos ({len(all_states)} frames)")

# Pad captions to match total state count
while len(all_captions) < len(all_states):
    all_captions.append(all_captions[-1] if all_captions else {})
captions_data = all_captions[:len(all_states)]

states = all_states
lead_car_data = all_lead_car
traffic_light_data = all_traffic_lights

# CoT arrays must match state count 1:1 (one entry per frame, same order)
assert len(lead_car_data) == len(states), f"lead_car_data ({len(lead_car_data)}) must match states ({len(states)})"
assert len(traffic_light_data) == len(states), f"traffic_light_data ({len(traffic_light_data)}) must match states ({len(states)})"

print(f"\n✓ Loaded {len(states)} frames from {len(video_ids_loaded)} videos")
print(f"✓ Loaded {len(captions_data)} captions")
if has_front_car:
    print(f"✓ Loaded {len(lead_car_data)} front_car frames")
if has_traffic_lights:
    print(f"✓ Loaded {len(traffic_light_data)} traffic_lights frames")

# Sample at EXTRACT_FPS (must match frame extraction)
EXTRACT_FPS = 2
FRAME_INTERVAL = 20 // EXTRACT_FPS  # 10 for 2Hz
states = states[::FRAME_INTERVAL]
captions_data = captions_data[::FRAME_INTERVAL]
lead_car_data = lead_car_data[::FRAME_INTERVAL]
traffic_light_data = traffic_light_data[::FRAME_INTERVAL]
assert len(lead_car_data) == len(states), f"After subsample: lead_car_data ({len(lead_car_data)}) must match states ({len(states)})"
assert len(traffic_light_data) == len(states), f"After subsample: traffic_light_data ({len(traffic_light_data)}) must match states ({len(states)})"
print(f"✓ Sampled at {EXTRACT_FPS}Hz: {len(states)} states, {len(captions_data)} captions")
if has_front_car or has_traffic_lights:
    print(f"  CoT: {len(lead_car_data)} lead_car, {len(traffic_light_data)} traffic_lights")

# IMPORTANT: Build image_files (in Cell 7-F) so len(image_files) == len(states), same order.
# Then create datasets: lead_car_data required; traffic_light_data optional (pass None for lead-car-only CoT)
train_dataset = CoVLADatasetPaper(states, captions_data, image_files, config, split="train",
    lead_car_data=lead_car_data, traffic_light_data=None)
val_dataset = CoVLADatasetPaper(states, captions_data, image_files, config, split="val",
    lead_car_data=lead_car_data, traffic_light_data=None)
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

# Build image_files to match states 1:1 (required: len(image_files) == len(states))
# States were subsampled at 2Hz in Cell 5-F (FRAME_INTERVAL = 10). Include only those frame indices.
image_files = []
for video_id in video_ids_loaded:
    video_frames_dir = f"{frames_dir}/{video_id}"
    if os.path.exists(video_frames_dir):
        for idx in range(0, 600, FRAME_INTERVAL):  # 0, 10, 20, ... 590 — same as states subsample
            path = os.path.join(video_frames_dir, f"{idx:04d}.png")
            if os.path.exists(path):
                image_files.append(path)
assert len(image_files) == len(states), f"image_files ({len(image_files)}) must match states ({len(states)})"

print(f"✓ Total frames available: {len(image_files)} (must match states for dataset)")
print(f"  From {len(video_ids_loaded)} videos, 2Hz subsample")
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
# Option A: Paper Model — Mistral 7B (RECOMMENDED for 24GB GPU)
# ~14GB FP16, batch_size 4-5, use gradient_accumulation for larger effective batch
# ============================================================
config = CoVLAConfig(
    device="cuda",
    model_size="paper",        # CLIP ViT-L + Mistral 7B
    batch_size=4,
    gradient_accumulation_steps=4,  # effective batch = 16
    learning_rate=2e-5,        # Cosine decay to 1/3 of this
    
    # Data augmentation (helps prevent overfitting)
    augment_trajectory=True,   # Add noise to GT trajectory
    augment_ego_state=True,    # Add noise to ego state
    augment_image=True,        # Color jitter
)

# ============================================================
# Option B: Mixtral 8x7B MoE (for 48GB GPU, e.g. A6000)
# ~26GB active FP16, better caption quality, batch_size 4-6
# ============================================================
# config = CoVLAConfig(
#     device="cuda",
#     model_size="mixtral",    # CLIP ViT-L + Mixtral 8x7B MoE
#     batch_size=4,
#     gradient_accumulation_steps=4,
# )

# ============================================================
# Option C: Lightweight Model (for free Colab T4 GPU)
# Works with ~8GB VRAM but caption quality is lower
# ============================================================
# config = CoVLAConfig(
#     device="cuda",
#     model_size="light",      # CLIP ViT-B + TinyLlama 1.1B
#     batch_size=8,
# )

# Create datasets: lead_car_data required; traffic_light_data optional (pass None for lead-car-only)
train_dataset = CoVLADatasetPaper(states, captions_data, image_files, config, split="train",
    lead_car_data=lead_car_data, traffic_light_data=None)
val_dataset = CoVLADatasetPaper(states, captions_data, image_files, config, split="val",
    lead_car_data=lead_car_data, traffic_light_data=None)

# Create model
model = CoVLAAgentPaper(config)

# Train (auto-saves after each epoch)
# Learning rate: cosine decay from 2e-5 to 6.7e-6
trainer = CoVLATrainerPaper(model, config)
history = trainer.train(train_dataset, val_dataset, num_epochs=4)

# Saves automatically:
# - covla_epoch_1.pt, covla_epoch_2.pt, ... (each epoch)
# - covla_best.pt (best ADE model)

# Manual save (optional)
# model.save_trainable("my_model.pt")  # ~50MB (only trainable weights)
```

### Cell 6c: Load Saved Model (skip training)
```python
from covla_agent_paper import load_model, CoVLAConfig, CoVLAAgentPaper

# Option 1: Simple load
model = load_model("covla_best.pt")

# Option 2: Load specific epoch
model = load_model("covla_epoch_5.pt", device="cuda")

# Option 3: Manual load (if you need custom config)
config = CoVLAConfig(device="cuda", model_size="paper")
model = CoVLAAgentPaper(config)
model.load_trainable("covla_best.pt")
```

### Cell 6c-alt: Continue training from checkpoint (e.g. 2 more epochs)
```python
from covla_agent_paper import CoVLAAgentPaper, CoVLATrainerPaper

# Set LR first so we continue from end of epoch 3 (don't restart at 2e-5)
config.learning_rate = 6.7e-6  # was eta_min after 3 epochs
model = CoVLAAgentPaper(config)
model.load_trainable("covla_best.pt")  # load weights only; config already set above
model.train()
trainer = CoVLATrainerPaper(model, config)
history = trainer.train(train_dataset, val_dataset, num_epochs=2)

# New checkpoints: covla_epoch_4.pt, covla_epoch_5.pt; covla_best.pt updated if ADE improves
```

### Model Comparison

| Setting | Vision | Language | VRAM (FP16) | Caption Quality | Approval |
|---------|--------|----------|------|-----------------|----------|
| `model_size="mixtral"` | CLIP ViT-L/14 | Mixtral 8x7B MoE | ~26GB active | ⭐⭐⭐⭐ Best | ✅ None |
| `model_size="paper"` | CLIP ViT-L/14 | Mistral 7B | ~14GB | ⭐⭐⭐ Great | ✅ None |
| `model_size="light"` | CLIP ViT-B/32 | TinyLlama 1.1B | ~2.2GB | ⭐ Basic | ✅ None |

### Saved Files

Training automatically saves efficient checkpoints (~50MB instead of ~5GB):

| File | Description |
|------|-------------|
| `covla_epoch_N.pt` | Checkpoint after epoch N |
| `covla_best.pt` | Best model (lowest validation ADE) |

Only trainable weights are saved (LoRA, projections, embeddings). Base models are reloaded from HuggingFace.

**Alternative 7B models** (change in config if needed):
- `mistralai/Mistral-7B-Instruct-v0.2` - Open, no approval (DEFAULT)
- `meta-llama/Llama-2-7b-hf` - Requires Meta approval
- `NousResearch/Llama-2-7b-hf` - Community mirror, no approval

### Memory Tips
- **A6000 (48GB)**: Use `model_size="mixtral"` for best quality, or `model_size="paper"` with large batch
- **A100 (40/80GB)**: Use `model_size="mixtral"` comfortably
- **24GB GPU (3090/4090)**: Use `model_size="paper"` with `gradient_accumulation_steps=4`
- **Free Colab T4 (16GB)**: Use `model_size="light"`

### Cell 7: Evaluate and Visualize
```python
# Plot training curves
plot_training_curves(history)

# Single prediction (generates caption → uses it for trajectory)
sample = val_dataset[0]
result = model.predict(
    sample['image'], 
    ego_state=sample['ego_state'],  # [vEgo/30, aEgo/5, steering/500] normalized
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

### Cell 10: Generate Evaluation Images and Video
```python
from covla_agent_paper import generate_eval_images

# Generate images with automatic video creation
generate_eval_images(
    model, 
    val_dataset, 
    output_dir="eval",      # Output directory
    num_frames=100,         # Number of frames to evaluate
    caption_mode="pred",    # "pred" or "gt"
    generate_video=True,    # Also create eval.mp4
    fps=3,                  # Video FPS
)
# Output: eval/0000.png, eval/0001.png, ..., eval/eval.mp4
```

**Output shows:**
- 🟢 Green: Ground truth trajectory
- 🔴 Red: Predicted trajectory  
- Left panel: Camera view with trajectory overlay
- Right panel: Bird's eye view
- Top: Speed, acceleration, steering, ADE/FDE metrics
- Bottom: GT caption (green) and predicted caption (red)

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

