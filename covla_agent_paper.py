"""
CoVLA-Agent: Exact Implementation from Paper
=============================================

Paper: "CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving"
https://arxiv.org/pdf/2408.10845

Section 4: Experiments
----------------------
- Dataset: 70% train, 15% val, 15% test
- Frames sampled at 2Hz
- Trajectory: 10 points (uniformly sampled from 60) for 3-second horizon
- Two tasks: Caption generation (CE) + Trajectory prediction (MSE)
- Loss: Equally weighted (0.5 * caption_loss + 0.5 * trajectory_loss)
- Metrics: ADE (Average Displacement Error), FDE (Final Displacement Error)

Results from paper:
- Predicted captions: ADE 0.955, FDE 2.239
- Ground truth captions: ADE 0.814, FDE 1.655
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np
import json
import os
# =============================================================================
# Navigation Commands (minimal set for VLA)
# =============================================================================

NAV_COMMANDS = ['LEFT', 'RIGHT', 'STRAIGHT']  # 3 basic commands
NAV_CMD_TO_IDX = {cmd: i for i, cmd in enumerate(NAV_COMMANDS)}

def get_nav_cmd(text: str) -> str:
    """
    Extract navigation command from caption text.
    Returns: 'LEFT', 'RIGHT', or 'STRAIGHT'
    
    Navigation semantics (like Google Maps):
    - LEFT/RIGHT = actual turn at intersection, exit ramp, etc.
    - STRAIGHT = continue on current road (including curves, lane follow)
    
    Note: "curve left/right" = following curved road = STRAIGHT (not a turn)
    """
    import re
    text_lower = text.lower()
    
    # LEFT: actual turns only (not curves - curves are lane following)
    # Matches: "turn left", "turning left", "left turn", "takes a left"
    if re.search(r'\bturn(?:ing|s)?\s+left|\bleft\s+turn|\btakes?\s+a\s+left', text_lower):
        return 'LEFT'
    
    # RIGHT: actual turns only
    if re.search(r'\bturn(?:ing|s)?\s+right|\bright\s+turn|\btakes?\s+a\s+right', text_lower):
        return 'RIGHT'
    
    # Default: STRAIGHT (includes curves, lane following, highway driving)
    return 'STRAIGHT'


# =============================================================================
# Ego State Normalization Constants
# =============================================================================

EGO_STATE_SCALES = {
    'vEgo': 30.0,           # Speed: 0-30 m/s → 0-1
    'aEgo': 5.0,            # Accel: -5 to 5 m/s² → -1 to 1
    'steeringAngleDeg': 500.0,  # Steering: -500 to 500° → -1 to 1
}

def normalize_ego_state(vEgo: float, aEgo: float = 0.0, steeringAngleDeg: float = 0.0) -> np.ndarray:
    """Normalize ego state values to [-1, 1] range."""
    return np.array([
        vEgo / EGO_STATE_SCALES['vEgo'],
        aEgo / EGO_STATE_SCALES['aEgo'],
        steeringAngleDeg / EGO_STATE_SCALES['steeringAngleDeg'],
    ], dtype=np.float32)

def denormalize_ego_state(ego_state: np.ndarray) -> dict:
    """Denormalize ego state array back to original units."""
    return {
        'vEgo': ego_state[0] * EGO_STATE_SCALES['vEgo'],
        'aEgo': ego_state[1] * EGO_STATE_SCALES['aEgo'],
        'steeringAngleDeg': ego_state[2] * EGO_STATE_SCALES['steeringAngleDeg'],
    }


# =============================================================================
# CoT R2 (EMMA-style: critical objects with BEV coordinates)
# =============================================================================

def format_r2_emma(lead_car: dict = None) -> str:
    """
    Format R2 (critical objects) in EMMA style. Lead car only; traffic lights skipped (add in next exp).
    EMMA: "Critical objects are on-road agents that can influence driving, with precise 3D/BEV coordinates."
    """
    parts = []
    if lead_car and lead_car.get('has_lead', False):
        x = lead_car.get('lead_x')
        y = lead_car.get('lead_y')
        if x is not None and y is not None:
            parts.append(f"vehicle at [{float(x):.1f}, {float(y):.1f}]")
        else:
            parts.append("vehicle ahead")
    if parts:
        return "Critical objects: " + "; ".join(parts)
    return "Critical objects: none"


def parse_r2_from_caption(caption: str) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Parse R2 (lead vehicle) from a caption string.
    Returns (has_lead, x, y). x,y are None if no lead or not parseable.
    """
    import re
    if not caption:
        return False, None, None
    # "Critical objects: none" or "... vehicle at [x, y] ..."
    if "Critical objects: none" in caption:
        return False, None, None
    match = re.search(r"vehicle at\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]", caption)
    if match:
        try:
            x, y = float(match.group(1)), float(match.group(2))
            return True, x, y
        except ValueError:
            return True, None, None
    if "vehicle at" in caption or "vehicle ahead" in caption:
        return True, None, None  # lead present but no coords
    return False, None, None


# =============================================================================
# Configuration (matching paper)
# =============================================================================

@dataclass
class CoVLAConfig:
    """
    Configuration matching paper's experiment setup.
    
    Paper model (requires ~24GB+ VRAM):
    - Vision: CLIP ViT-L/14
    - Language: Llama-2 7B
    - Speed embedding MLP
    - Trajectory query tokens
    
    Set model_size="paper" for exact paper architecture (Mistral 7B).
    Set model_size="mixtral" for Mixtral 8x7B MoE (48GB GPU).
    Set model_size="light" for lightweight version (free Colab).
    """
    
    # Model selection: "light", "paper", or "mixtral"
    model_size: str = "paper"  # "light" (~8GB), "paper" (~14GB FP16), "mixtral" (~26GB FP16 active)
    
    # Model options
    # light:   CLIP ViT-B/32 + TinyLlama 1.1B   (~8GB VRAM, free Colab)
    # paper:   CLIP ViT-L/14 + Mistral 7B        (~14GB FP16, 24GB GPU)
    # mixtral: CLIP ViT-L/14 + Mixtral 8x7B MoE  (~26GB active FP16, 48GB GPU)
    vision_encoder_light: str = "openai/clip-vit-base-patch32"
    language_model_light: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    vision_encoder_paper: str = "openai/clip-vit-large-patch14"
    language_model_paper: str = "mistralai/Mistral-7B-Instruct-v0.2"
    vision_encoder_mixtral: str = "openai/clip-vit-large-patch14"
    language_model_mixtral: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    # Trajectory (paper uses 10 points, sampled from 60)
    trajectory_points: int = 10  # Paper: "10 points uniformly sampled"
    trajectory_dim: int = 3  # x, y, z
    trajectory_horizon: float = 3.0  # seconds
    
    # Ego state embedding (paper includes ego vehicle speed)
    use_extended_ego_state: bool = False  # Use [vEgo, aEgo, steering] instead of just speed
    ego_state_dim: int = 3  # vEgo, aEgo, steeringAngleDeg
    
    # Navigation command input (high-level instruction like Google Maps)
    use_nav_cmd: bool = False  # Add nav command (LEFT/RIGHT/STRAIGHT) as input
    num_nav_cmds: int = 3  # Number of navigation commands
    
    # Temporal modeling (multi-frame input, always enabled)
    num_history_frames: int = 2  # Number of history frames (at 2Hz, 2 frames = 1 second of history)
    
    # Training
    batch_size: int = 8
    eval_batch_size: int = 16  # Eval has no gradients, can use larger batch
    learning_rate: float = 2e-5
    num_epochs: int = 10
    
    # Loss weights (paper: equally weighted)
    caption_weight: float = 0.5
    trajectory_weight: float = 0.5
    smoothing_weight: float = 0.1  # Smoothing loss to reduce trajectory wobble
    
    # Data split (80/20, no test)
    train_ratio: float = 0.80
    val_ratio: float = 0.20
    test_ratio: float = 0.0
    
    # Data subsampling (to cover more videos with less compute)
    # sample_stride=1: use all samples (default)
    # sample_stride=2: use every 2nd sample (2x video coverage, same compute)
    # sample_stride=3: use every 3rd sample (3x video coverage, same compute)
    sample_stride: int = 1
    
    # Performance optimization
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps (simulate larger batch)
    quantize: str = "none"  # "none", "8bit", or "4bit" — reduces LLM memory, enables larger batch size
    
    # Extra R2 loss: CE on tokens before the first \n in caption (= R2 prefix)
    caption_r2_weight: float = 1.0
    # Val R2 metrics (generate_caption): 1-based epoch index; skip before this to save time.
    # R2 is always computed on the last epoch as well (so 1-epoch runs still get R2).
    eval_r2_from_epoch: int = 2
    
    # Trajectory-only distillation: add distill_traj_weight * MSE(student pred, frozen teacher pred).
    # Teacher forward uses GT captions for conditioning (same as student). Works across different LLM/tokenizers.
    distill_traj_weight: float = 0.0  # 0 = off; try ~0.2–1.0 with teacher=CoVLAAgentPaper(...)
    # Trajectory-query hidden distillation: MSE(projector(student LLM hiddens), teacher hiddens) at the 10 query
    # positions (before Trajectory MLP). Student gets a Linear(student_llm_dim -> distill_teacher_llm_dim).
    distill_traj_feat_weight: float = 0.0
    distill_teacher_llm_dim: Optional[int] = None  # e.g. 4096 for Mistral-7B; required if distill_traj_feat_weight > 0
    
    # Data augmentation (only applied during training)
    augment_trajectory: bool = True  # Add noise to GT trajectory
    augment_ego_state: bool = True   # Add noise to ego state
    augment_image: bool = True       # Color jitter on images
    trajectory_noise_std: float = 0.05  # meters
    ego_state_noise_std: float = 0.02   # relative (2%)
    
    # Frame sampling (paper: 2Hz)
    frame_sample_rate: int = 2  # Hz
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Image (CLIP native sizes: ViT-L/14=224, ViT-B/32=224)
    image_size: int = 224
    
    # LoRA
    use_lora: bool = True
    lora_rank: int = 16
    
    # Diffusion trajectory head (optional; replaces deterministic TrajectoryMLP)
    use_diffusion_trajectory: bool = False
    diffusion_num_timesteps: int = 50
    diffusion_hidden_dim: int = 256
    diffusion_cond_dim: int = 256  # projected from LLM dim for the noise MLP
    diffusion_traj_scale: float = 25.0  # divide coords by this before diffusion (rough meter scale)
    diffusion_loss_bins: int = 5  # log noise MSE averaged over t bins (train)
    diffusion_eval_num_samples: int = 8  # default K for eval image fan (set generate_eval_images diffusion_fan_k)
    diffusion_viz_num_samples: int = 1  # training viz (diffusion): N independent sample() curves + always Pred(x0_hat); N=0 → x0_hat only
    
    @property
    def vision_encoder(self) -> str:
        return getattr(self, f"vision_encoder_{self.model_size}")
    
    @property
    def language_model(self) -> str:
        return getattr(self, f"language_model_{self.model_size}")


# =============================================================================
# Dataset (matching paper's preprocessing)
# =============================================================================

class CoVLADatasetPaper(Dataset):
    """
    Dataset following paper's Section 4.1 preprocessing:
    - Frames sampled at 2Hz
    - 10 trajectory points (uniformly sampled from 60)
    - Excludes frames without complete 3-second trajectory
    - Optional CoT R2 (EMMA-style): critical objects with BEV coords
    """
    
    def __init__(
        self,
        states_data: List[Dict],
        captions_data: List[Dict],
        image_files: List[str],
        config: CoVLAConfig,
        split: str = "train",  # "train", "val", or "test"
        lead_car_data: Optional[List[Dict]] = None,   # For CoT R2 (aligned with states_data indices)
        traffic_light_data: Optional[List[Dict]] = None,
    ):
        self.config = config
        self.split = split
        assert lead_car_data is not None, "lead_car_data required (CoT: lead car first; traffic_light_data optional for later)"
        
        # Data is already sampled at 2Hz (0.5s between consecutive frames)
        # sample_interval is kept for backward compatibility but should be 1 for 2Hz data
        sample_interval = 1  # Data is already at 2Hz
        
        # Filter and prepare samples with logging
        self.samples = []
        self.physics_mismatch_indices = []  # Store indices for debugging
        filter_counts = {
            'total_candidates': 0,
            'insufficient_history': 0,
            'cross_video_boundary': 0,
            'incomplete_trajectory': 0,
            'absolute_too_large': 0,
            'delta_too_large': 0,
            'lateral_delta_too_large': 0,
            'physics_mismatch': 0,
            'no_imu_data': 0,
            'passed': 0,
        }
        
        for i in range(0, len(states_data), sample_interval):
            if i >= len(image_files):
                break
            
            filter_counts['total_candidates'] += 1
            
            # Check if we have enough history frames (at 2Hz, each index is 0.5s apart)
            # Need frames at i, i-1, i-2, ... for temporal model
            history_indices = [i - k for k in range(config.num_history_frames + 1)]
            if any(idx < 0 or idx >= len(image_files) for idx in history_indices):
                filter_counts['insufficient_history'] += 1
                continue
            
            # Check all frames are from the same video (same parent directory = same video ID)
            frame_paths = [image_files[idx] for idx in history_indices]
            video_dirs = [os.path.dirname(p) for p in frame_paths]
            if len(set(video_dirs)) > 1:
                filter_counts['cross_video_boundary'] += 1
                continue
            
            state = states_data[i]
            trajectory = state.get('trajectory', [])
            
            # Paper: "excluding those lacking complete trajectory data for subsequent 3 seconds"
            if len(trajectory) < 60:
                filter_counts['incomplete_trajectory'] += 1
                continue
            
            # Uniformly sample 10 points from 60 (paper specification)
            traj_indices = np.linspace(0, 59, config.trajectory_points, dtype=int)
            sampled_trajectory = [trajectory[j] for j in traj_indices]
            
            # Filter corrupted trajectories
            traj_array = np.array(sampled_trajectory)
            
            # Check 1: Absolute values (should be <200m in 3s horizon)
            if np.any(np.abs(traj_array) > 200):
                filter_counts['absolute_too_large'] += 1
                continue
            
            # Check 2: Delta between consecutive points
            deltas = np.diff(traj_array, axis=0)  # (9, 3)
            if np.any(np.abs(deltas) > 20):  # 20m per interval = ~67 m/s = 240 km/h
                filter_counts['delta_too_large'] += 1
                continue
            
            # Check 3: Lateral (y) delta - cars can't move sideways fast
            if np.any(np.abs(deltas[:, 1]) > 5):  # Max 5m lateral per interval
                filter_counts['lateral_delta_too_large'] += 1
                continue
            
            # Check 4: Physics-based trajectory validation using IMU yaw rate
            final_x = traj_array[-1, 0]  # Actual forward distance (meters)
            final_y = traj_array[-1, 1]  # Actual lateral distance (meters, signed)
            
            # Note: final_x can be negative for sharp turns (U-turn, 90° turn)
            # The physics simulation will validate if the trajectory matches sensor data
            angular_vel = state.get('angular_velocities_calib', None)
            trajectory_physics = None  # Simulated trajectory from physics model
            
            if angular_vel is not None and len(angular_vel) >= 3:
                raw_speed = state['vEgo']  # m/s
                raw_accel = state.get('aEgo', 0.0)  # m/s²
                # Yaw rate from IMU: [2] = rotation around vertical axis
                # In standard coords: negative yaw_rate = turning right (clockwise from above)
                # Our Y+ = right, so we negate to get correct lateral direction
                yaw_rate = -angular_vel[2]
                
                # Simulate trajectory over 3 seconds, sample 10 points to match GT
                dt = 0.1
                sim_x, sim_y, sim_heading = 0.0, 0.0, 0.0
                sim_v = raw_speed
                sim_points = []
                sim_1s = None  # Position at 1 second for validation
                
                for step in range(30):  # 30 steps × 0.1s = 3s
                    sim_v = max(0, sim_v + raw_accel * dt)
                    sim_heading += yaw_rate * dt
                    sim_x += sim_v * np.cos(sim_heading) * dt
                    sim_y += sim_v * np.sin(sim_heading) * dt
                    
                    # Save position at 1 second (step 10)
                    if step == 9:
                        sim_1s = (sim_x, sim_y)
                    
                    # Sample at same intervals as GT (every 3rd step for 10 points)
                    if (step + 1) % 3 == 0:
                        sim_points.append([sim_x, sim_y, 0.0])  # z=0 for ground plane
                
                trajectory_physics = sim_points  # 10 points [(x, y, z), ...]
                
                # Compare at 1 second (more reliable than 3s where errors accumulate)
                # GT point 3 ≈ 1 second (10 points over 3s, so point 3 = 0.9s)
                gt_1s = traj_array[3]  # (x, y, z) at ~1 second
                
                # Tolerance: min 3m base + 50% of distance (relaxed)
                dist_1s = np.sqrt(sim_1s[0]**2 + sim_1s[1]**2) if sim_1s else 0
                tolerance = max(3.0, dist_1s * 0.5)
                
                error_1s = np.sqrt((gt_1s[0] - sim_1s[0])**2 + (gt_1s[1] - sim_1s[1])**2) if sim_1s else 0
                gt_dist_1s = np.sqrt(gt_1s[0]**2 + gt_1s[1]**2)
                
                if error_1s > tolerance and gt_dist_1s > 2:  # Only check if moved >2m in 1s
                    filter_counts['physics_mismatch'] += 1
                    self.physics_mismatch_indices.append(i)
                    continue
            else:
                filter_counts['no_imu_data'] += 1
            
            # Get caption
            caption_idx = min(i // sample_interval, len(captions_data) - 1)
            caption = captions_data[caption_idx] if captions_data else {}
            caption_text = caption.get('rich_caption', caption.get('plain_caption', ''))
            
            # CoT R2 (EMMA-style): R2 first, then caption. Lead car only; traffic lights skipped (add in next exp).
            assert i < len(lead_car_data), f"lead_car_data index {i} >= len {len(lead_car_data)}"
            lead_car = lead_car_data[i]
            r2_text = format_r2_emma(lead_car)
            full_caption = f"{r2_text}\n{caption_text}"
            
            # Get speed (required field - called 'vEgo' in dataset)
            if 'vEgo' not in state:
                raise KeyError(f"'vEgo' not found in state at index {i}. Available keys: {list(state.keys())}")
            
            # Extended ego state: [vEgo, aEgo, steeringAngleDeg] normalized
            ego_state = normalize_ego_state(
                state['vEgo'],
                state.get('aEgo', 0.0),
                state.get('steeringAngleDeg', 0.0),
            )
            
            # Extract navigation command from caption
            nav_cmd = get_nav_cmd(caption_text)
            
            filter_counts['passed'] += 1
            
            # Collect frame paths (oldest to newest): [t-2, t-1, t]
            image_paths = [
                image_files[i - k]
                for k in range(config.num_history_frames, -1, -1)  # 2, 1, 0
            ]
            
            self.samples.append({
                'image_paths': image_paths,  # All frames: [oldest, ..., current]
                'trajectory': sampled_trajectory,
                'trajectory_physics': trajectory_physics,  # Simulated from IMU (10 points or None)
                'caption': full_caption,
                'ego_state': ego_state,  # [vEgo/30, aEgo/5, steering/500] normalized
                'nav_cmd': nav_cmd,  # 'LEFT', 'RIGHT', or 'STRAIGHT'
                'extrinsic_matrix': state['extrinsic_matrix'],
                'intrinsic_matrix': state['intrinsic_matrix'],
            })
        
        # Print filter summary
        total = filter_counts['total_candidates']
        passed = filter_counts['passed']
        filtered = total - passed
        print(f"\n📊 Data filtering summary:")
        print(f"   Total candidates: {total}")
        print(f"   ├─ Insufficient history: {filter_counts['insufficient_history']}")
        print(f"   ├─ Cross video boundary: {filter_counts['cross_video_boundary']}")
        print(f"   ├─ Incomplete trajectory: {filter_counts['incomplete_trajectory']}")
        print(f"   ├─ Absolute value >200m: {filter_counts['absolute_too_large']}")
        print(f"   ├─ Delta >20m: {filter_counts['delta_too_large']}")
        print(f"   ├─ Lateral delta >5m: {filter_counts['lateral_delta_too_large']}")
        print(f"   ├─ Physics mismatch: {filter_counts['physics_mismatch']}")
        print(f"   └─ No IMU data (kept): {filter_counts['no_imu_data']}")
        print(f"   ✓ Passed: {passed} ({100*passed/total:.1f}%)")
        print(f"   ✗ Filtered: {filtered} ({100*filtered/total:.1f}%)")
        num_frames = config.num_history_frames + 1
        tokens_per_frame = 50 if 'base' in config.vision_encoder else 257
        print(f"   📹 Temporal: {num_frames} frames/sample ({num_frames} × {tokens_per_frame} = {num_frames * tokens_per_frame} tokens)")
        print(f"   📝 CoT R2 (EMMA-style): R2 then caption")
        
        # Subsample to cover more videos with less compute
        if config.sample_stride > 1:
            original_count = len(self.samples)
            self.samples = self.samples[::config.sample_stride]
            print(f"   📉 Subsampled: {original_count} → {len(self.samples)} (stride={config.sample_stride})")
        
        # Split data (80/20 train/val)
        n = len(self.samples)
        train_end = int(n * config.train_ratio)
        val_end = train_end + int(n * config.val_ratio)
        
        if split == "train":
            self.samples = self.samples[:train_end]
        elif split == "val":
            self.samples = self.samples[train_end:val_end]
        else:  # test
            self.samples = self.samples[val_end:]
        
        # Image transforms: add color jitter for training augmentation
        if split == "train" and config.augment_image:
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        print(f"✓ {split.upper()} set: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load all frames (oldest to newest): [t-2, t-1, t]
        images = []
        for img_path in sample['image_paths']:
            try:
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
            except Exception:
                img = torch.zeros(3, self.config.image_size, self.config.image_size)
            images.append(img)
        
        # Stack: (num_frames, C, H, W) - oldest to newest
        images = torch.stack(images, dim=0)
        
        # Trajectory: (10, 3)
        trajectory = torch.tensor(sample['trajectory'], dtype=torch.float32)
        
        # Ego state: [vEgo/30, aEgo/5, steering/500] normalized
        ego_state = torch.tensor(sample['ego_state'], dtype=torch.float32)
        
        # Data augmentation (training only)
        if self.split == "train":
            # Trajectory noise (small Gaussian, ~5cm std)
            if self.config.augment_trajectory:
                trajectory = trajectory + torch.randn_like(trajectory) * self.config.trajectory_noise_std
            
            # Ego state noise (relative, ~2%)
            if self.config.augment_ego_state:
                ego_state = ego_state * (1 + torch.randn_like(ego_state) * self.config.ego_state_noise_std)
        # Navigation command as index (tensor for DataLoader)
        nav_cmd = sample.get('nav_cmd', 'STRAIGHT')
        nav_cmd_idx = torch.tensor(NAV_CMD_TO_IDX.get(nav_cmd, NAV_CMD_TO_IDX['STRAIGHT']), dtype=torch.long)
        
        # Physics-based simulated trajectory (for visualization/debugging)
        traj_physics = sample.get('trajectory_physics')
        if traj_physics is not None:
            traj_physics = torch.tensor(traj_physics, dtype=torch.float32)  # (10, 3)
        
        return {
            'images': images,  # (num_frames, C, H, W) - all frames [t-2, t-1, t]
            'trajectory': trajectory,
            'trajectory_physics': traj_physics,  # (10, 3) or None - simulated from IMU
            'caption': sample['caption'],
            'ego_state': ego_state,  # (3,) [vEgo/30, aEgo/5, steering/500] normalized
            'nav_cmd': nav_cmd,  # string: 'LEFT', 'RIGHT', 'STRAIGHT'
            'nav_cmd_idx': nav_cmd_idx,  # tensor: 0, 1, or 2
            'extrinsic_matrix': sample.get('extrinsic_matrix'),
            'intrinsic_matrix': sample.get('intrinsic_matrix'),
            'image_paths': sample['image_paths'],  # All frame paths [t-2, t-1, t] for visualization
        }


# =============================================================================
# Trajectory MLP (paper specification)
# =============================================================================

class TrajectoryMLP(nn.Module):
    """
    MLP for trajectory prediction (per-query version).
    
    Paper Figure 5: Each trajectory query token → MLP → (x, y, z)
    """
    
    def __init__(self, input_dim: int, num_points: int = 10, coord_dim: int = 3):
        super().__init__()
        self.num_points = num_points
        self.coord_dim = coord_dim
        
        # Per-query MLP: each query token → (x, y, z)
        # Small hidden dim to avoid overfitting with limited data
        hidden_dim = 64
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(hidden_dim, coord_dim),  # Output 3 coords per query
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Trajectory query outputs (batch, num_points, hidden_dim)
        Returns:
            trajectory: (batch, num_points, coord_dim)
        """
        # Apply MLP to each query token
        return self.mlp(x)  # (batch, num_points, coord_dim)


def _extract_ddpm(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Gather a[t] for each batch item and reshape to broadcast with x."""
    b = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class TrajectoryDiffusionHead(nn.Module):
    """
    Minimal conditional DDPM on flattened trajectory (noise prediction).
    Conditioning: pooled trajectory-query LLM states (mean over queries).
    """

    def __init__(
        self,
        cond_dim: int,
        num_points: int,
        coord_dim: int,
        num_timesteps: int,
        hidden_dim: int,
        cond_proj_dim: int,
        traj_scale: float,
    ):
        super().__init__()
        self.num_points = num_points
        self.coord_dim = coord_dim
        self.flat_dim = num_points * coord_dim
        self.num_timesteps = num_timesteps
        self.traj_scale = traj_scale

        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_acp", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

        te = 64
        self.time_embed = nn.Embedding(num_timesteps, te)
        self.cond_proj = nn.Linear(cond_dim, cond_proj_dim)
        in_dim = self.flat_dim + cond_proj_dim + te
        self.eps_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.flat_dim),
        )

    def _predict_eps(self, x_flat: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x_flat: (B, flat_dim), t: (B,) long, cond: (B, cond_dim)"""
        te = self.time_embed(t)
        c = self.cond_proj(cond)
        h = torch.cat([x_flat, c, te], dim=-1)
        return self.eps_net(h)

    def training_loss_and_x0_hat(
        self, x0_phys: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x0_phys: (B, P, 3) in dataset units.
        Returns diffusion loss on noise, and x0_hat (B, P, 3) for metrics / smoothing.
        """
        device = x0_phys.device
        B = x0_phys.shape[0]
        x0 = (x0_phys.reshape(B, -1) / self.traj_scale).float()
        t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        sqrt_acp = _extract_ddpm(self.sqrt_acp, t, x0.shape)
        sqrt_om = _extract_ddpm(self.sqrt_one_minus_acp, t, x0.shape)
        x_t = sqrt_acp * x0 + sqrt_om * noise
        eps_pred = self._predict_eps(x_t, t, cond.float())
        loss = F.mse_loss(eps_pred, noise)
        mse_per = ((eps_pred - noise) ** 2).mean(dim=1).detach()  # (B,) for bin logging
        x0_hat = (x_t - sqrt_om * eps_pred) / (sqrt_acp + 1e-8)
        x0_hat = x0_hat.view(B, self.num_points, self.coord_dim) * self.traj_scale
        return loss, x0_hat, t, mse_per

    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        """DDPM reverse; cond: (B, cond_dim). Returns (B, P, 3) physical units."""
        self.eval()
        device = cond.device
        B = cond.shape[0]
        x = torch.randn(B, self.flat_dim, device=device, dtype=torch.float32)
        cond_f = cond.float()
        for ti in reversed(range(self.num_timesteps)):
            t = torch.full((B,), ti, device=device, dtype=torch.long)
            eps = self._predict_eps(x, t, cond_f)
            beta_t = _extract_ddpm(self.betas, t, x.shape)
            sqrt_om_ab = _extract_ddpm(self.sqrt_one_minus_acp, t, x.shape)
            sr = _extract_ddpm(self.sqrt_recip_alphas, t, x.shape)
            mean = sr * (x - beta_t / sqrt_om_ab * eps)
            if ti > 0:
                noise = torch.randn_like(x)
                var = _extract_ddpm(self.posterior_variance, t, x.shape)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean
        x = x.view(B, self.num_points, self.coord_dim) * self.traj_scale
        return x


# =============================================================================
# CoVLA-Agent (exact paper architecture)
# =============================================================================

class CoVLAAgentPaper(nn.Module):
    """
    CoVLA-Agent following paper's architecture exactly.
    
    Architecture:
    - Vision Encoder: Processes input frames
    - Language Model: Generates traffic scene descriptions
    - Trajectory MLP: Outputs 10 (x, y, z) coordinates
    
    Training:
    - Task 1: Traffic Scene Description (Cross-Entropy loss)
    - Task 2: Trajectory Prediction (MSE loss)
    - Combined loss: 0.5 * CE + 0.5 * MSE
    """
    
    def __init__(self, config: CoVLAConfig):
        super().__init__()
        self.config = config
        
        # Vision Encoder
        print("Loading Vision Encoder...")
        from transformers import CLIPModel, CLIPProcessor
        self.vision_encoder = CLIPModel.from_pretrained(config.vision_encoder)
        self.vision_processor = CLIPProcessor.from_pretrained(config.vision_encoder)
        
        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Language Model
        print("Loading Language Model...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        load_kwargs = {"torch_dtype": torch.float16 if config.device == "cuda" else torch.float32}
        if config.quantize == "8bit":
            load_kwargs["load_in_8bit"] = True
            print("  Loading LLM in 8-bit (QLoRA) — ~50% less VRAM")
        elif config.quantize == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            print("  Loading LLM in 4-bit (QLoRA NF4) — ~75% less VRAM")
        # MoE models (Mixtral) need device_map to handle expert routing across memory
        if config.model_size == "mixtral":
            load_kwargs["device_map"] = "auto"
            print("  Loading MoE model with device_map='auto'")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.language_model,
            **load_kwargs,
        )
        
        # Prepare quantized model for training (freeze base, enable gradient on adapters)
        if config.quantize != "none":
            from peft import prepare_model_for_kbit_training
            self.language_model = prepare_model_for_kbit_training(self.language_model)
        
        # Apply LoRA for efficient fine-tuning
        if config.use_lora:
            self._apply_lora()
        
        # Dimensions
        # Use hidden_size from vision model (patch tokens), not projection_dim (CLS only)
        self.vision_dim = self.vision_encoder.config.vision_config.hidden_size  # 768 (base) or 1024 (large)
        self.llm_dim = self.language_model.config.hidden_size  # 2048 (TinyLlama) or 4096 (Llama-2)
        
        # Vision projection to LLM space (projects each patch token)
        self.vision_projection = nn.Linear(self.vision_dim, self.llm_dim)
        
        # Trajectory query tokens (paper: learnable queries for trajectory prediction)
        self.num_trajectory_queries = config.trajectory_points  # 10 queries for 10 waypoints
        self.trajectory_queries = nn.Parameter(
            torch.randn(1, self.num_trajectory_queries, self.llm_dim) * 0.02
        )
        
        # Ego state embedding (paper: embeds ego vehicle speed)
        # Input dim: 1 for speed only, or ego_state_dim for extended state
        input_dim = config.ego_state_dim if config.use_extended_ego_state else 1
        self.speed_embedding = nn.Linear(input_dim, self.llm_dim)
        self.use_extended_ego_state = config.use_extended_ego_state
        
        # Navigation command embedding (optional)
        self.use_nav_cmd = config.use_nav_cmd
        if config.use_nav_cmd:
            self.nav_cmd_embedding = nn.Embedding(config.num_nav_cmds, self.llm_dim)
        
        # Temporal modeling (multi-frame input via concatenation)
        self.num_frames = config.num_history_frames + 1  # e.g., 3 frames for 2 history
        # Concat fusion: no extra params needed - just concatenate all frame tokens
        
        # Trajectory head: paper MLP, or optional conditional DDPM on flattened path
        self.use_diffusion_trajectory = config.use_diffusion_trajectory
        if self.use_diffusion_trajectory:
            self.traj_diffusion = TrajectoryDiffusionHead(
                cond_dim=self.llm_dim,
                num_points=config.trajectory_points,
                coord_dim=config.trajectory_dim,
                num_timesteps=config.diffusion_num_timesteps,
                hidden_dim=config.diffusion_hidden_dim,
                cond_proj_dim=config.diffusion_cond_dim,
                traj_scale=config.diffusion_traj_scale,
            )
            self.trajectory_mlp = None
        else:
            self.traj_diffusion = None
            self.trajectory_mlp = TrajectoryMLP(
                input_dim=self.llm_dim,
                num_points=config.trajectory_points,
                coord_dim=config.trajectory_dim,
            )
        
        # Optional projector for trajectory-query feature distillation (student hidden -> teacher hidden size)
        self.traj_feat_projector: Optional[nn.Linear] = None
        if config.distill_traj_feat_weight > 0:
            if config.distill_teacher_llm_dim is None:
                raise ValueError(
                    "distill_traj_feat_weight > 0 requires distill_teacher_llm_dim (teacher language_model.config.hidden_size)"
                )
            self.traj_feat_projector = nn.Linear(self.llm_dim, config.distill_teacher_llm_dim)
        
        # Loss weights
        self.smoothing_weight = config.smoothing_weight
        
        # Print model info
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_type = config.model_size.upper()
        print(f"✓ CoVLA-Agent initialized ({model_type})")
        print(f"  Vision: {config.vision_encoder}")
        print(f"  Language: {config.language_model}")
        print(f"  Ego state: {'extended (vEgo, aEgo, steering)' if config.use_extended_ego_state else 'speed only'}")
        print(f"  Nav command: {'enabled (LEFT/RIGHT/STRAIGHT)' if config.use_nav_cmd else 'disabled'}")
        print(f"  Temporal: {self.num_frames} frames (concat)")
        if self.use_diffusion_trajectory:
            print(
                f"  Trajectory: diffusion DDPM (T={config.diffusion_num_timesteps}, "
                f"scale={config.diffusion_traj_scale})"
            )
        else:
            print("  Trajectory: MLP (paper)")
        print(f"  Total params: {total:,}")
        print(f"  Trainable: {trainable:,}")
    
    def _apply_lora(self):
        """Apply LoRA to language model. Targets attention for standard models, attention + expert gates for MoE."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            
            target_modules = ["q_proj", "v_proj"]
            
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_rank * 2,
                target_modules=target_modules,
                lora_dropout=0.1,
                task_type=TaskType.CAUSAL_LM,
            )
            self.language_model = get_peft_model(self.language_model, lora_config)
            print(f"✓ LoRA applied (targets: {target_modules})")
        except ImportError:
            print("⚠ PEFT not installed, training full model")
    
    def save_trainable(self, path: str = "covla_trainable.pt"):
        """
        Save only trainable components (efficient - ~50MB instead of ~5GB).
        
        Saves: LoRA adapters, vision_projection, speed_embedding, 
               trajectory_queries, trajectory_mlp or traj_diffusion
        """
        trainable_state = {
            'vision_projection': self.vision_projection.state_dict(),
            'speed_embedding': self.speed_embedding.state_dict(),
            'trajectory_queries': self.trajectory_queries.data,
            'config': self.config,
        }
        if self.use_diffusion_trajectory:
            trainable_state['traj_diffusion'] = self.traj_diffusion.state_dict()
        else:
            trainable_state['trajectory_mlp'] = self.trajectory_mlp.state_dict()
        
        if self.traj_feat_projector is not None:
            trainable_state['traj_feat_projector'] = self.traj_feat_projector.state_dict()
        
        # Save nav_cmd_embedding if used
        if self.use_nav_cmd:
            trainable_state['nav_cmd_embedding'] = self.nav_cmd_embedding.state_dict()
        
        # Save LoRA weights if using PEFT
        if hasattr(self.language_model, 'peft_config'):
            trainable_state['lora'] = {
                k: v for k, v in self.language_model.state_dict().items() 
                if 'lora' in k.lower()
            }
        
        torch.save(trainable_state, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✓ Saved trainable weights to: {path} ({size_mb:.1f} MB)")
    
    def load_trainable(self, path: str = "covla_trainable.pt"):
        """
        Load only trainable components.
        
        Usage:
            model = CoVLAAgentPaper(config)  # Creates fresh base models
            model.load_trainable("covla_trainable.pt")  # Loads trained weights
        """
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.vision_projection.load_state_dict(checkpoint['vision_projection'])
        self.speed_embedding.load_state_dict(checkpoint['speed_embedding'])
        self.trajectory_queries.data = checkpoint['trajectory_queries'].to(self.config.device)
        if self.use_diffusion_trajectory:
            if 'traj_diffusion' not in checkpoint:
                raise ValueError(
                    "Checkpoint has no traj_diffusion. Use a checkpoint trained with "
                    "use_diffusion_trajectory=True, or set use_diffusion_trajectory=False for MLP ckpts."
                )
            self.traj_diffusion.load_state_dict(checkpoint['traj_diffusion'])
        else:
            self.trajectory_mlp.load_state_dict(checkpoint['trajectory_mlp'])
        
        if 'traj_feat_projector' in checkpoint and self.traj_feat_projector is not None:
            self.traj_feat_projector.load_state_dict(checkpoint['traj_feat_projector'])
        
        # Load nav_cmd_embedding if present
        if 'nav_cmd_embedding' in checkpoint and self.use_nav_cmd:
            self.nav_cmd_embedding.load_state_dict(checkpoint['nav_cmd_embedding'])
        
        # Load LoRA weights (strict=False so quant buffers like .absmax/.quant_map from 4bit are OK if not in current model)
        if 'lora' in checkpoint and hasattr(self.language_model, 'peft_config'):
            lora_state = checkpoint['lora']
            result = self.language_model.load_state_dict(lora_state, strict=False)
            if result.missing_keys:
                print(f"  Missing keys (kept as-is): {len(result.missing_keys)}")
            if result.unexpected_keys:
                print(f"  Unexpected keys (ignored): {len(result.unexpected_keys)}")
        
        # Move entire model to device
        self.to(self.config.device)
        
        print(f"✓ Loaded trainable weights from: {path}")
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using vision encoder.
        Returns all patch tokens (not just CLS) for richer visual representation.
        
        Args:
            images: (batch, C, H, W) single frame per sample
        Returns:
            patch_tokens: (batch, num_patches+1, vision_dim)
        """
        device = images.device
        
        # Convert to PIL for processor
        pil_images = []
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        
        for img in images:
            img_denorm = (img * std + mean).clamp(0, 1)
            img_np = (img_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        
        inputs = self.vision_processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get all patch tokens (LLaVA-style), not just CLS
            vision_outputs = self.vision_encoder.vision_model(pixel_values=inputs['pixel_values'])
            # last_hidden_state: (batch, num_patches+1, hidden_dim)
            # For ViT-L/14 with 224x224: (batch, 257, 1024)
            # For ViT-B/32 with 224x224: (batch, 50, 768)
            patch_tokens = vision_outputs.last_hidden_state
        
        return patch_tokens  # (batch, num_patches+1, vision_dim)
    
    def encode_temporal_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode multiple frames and concatenate them.
        
        Args:
            images: (batch, num_frames, C, H, W) multiple frames per sample
        Returns:
            fused_features: (batch, num_frames * num_patches, llm_dim) concatenated features
        """
        batch_size, num_frames, C, H, W = images.shape
        
        # Flatten batch and frames for efficient encoding
        flat_images = images.view(batch_size * num_frames, C, H, W)
        
        # Encode all frames at once
        patch_tokens = self.encode_image(flat_images)  # (B*T, num_patches, vision_dim)
        num_patches = patch_tokens.shape[1]
        
        # Reshape back: (batch, num_frames, num_patches, vision_dim)
        patch_tokens = patch_tokens.view(batch_size, num_frames, num_patches, -1)
        
        # Project to LLM space
        vision_embeds = self.vision_projection(patch_tokens)  # (B, T, P, llm_dim)
        
        # Concatenate all frame tokens: (batch, num_frames * num_patches, llm_dim)
        # Order: [frame_t-2, frame_t-1, frame_t] (oldest to newest)
        fused_features = vision_embeds.view(batch_size, num_frames * num_patches, -1)
        
        return fused_features
    
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[List[str]] = None,
        trajectories: Optional[torch.Tensor] = None,
        ego_state: torch.Tensor = None,
        nav_cmd_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both training and inference.
        
        Args:
            images: (batch, num_frames, C, H, W) multi-frame input
            captions: Captions for sequence (GT during training, predicted/GT during inference)
            trajectories: Ground truth trajectories (batch, 10, 3) - only for training
            ego_state: (batch, D) where D=1 for speed only, D=3 for extended [vEgo, aEgo, steering]
            nav_cmd_idx: (batch,) navigation command indices (0=LEFT, 1=RIGHT, 2=STRAIGHT)
        
        Training: captions = GT captions (for both trajectory conditioning and caption loss)
        Inference: captions = predicted (caption_mode="pred") or GT (caption_mode="gt")
        
        Returns:
            Dictionary with predictions and losses
        """
        device = images.device
        batch_size = images.shape[0]
        
        # Encode multi-frame input: (batch, num_frames, C, H, W)
        vision_embeds = self.encode_temporal_images(images)  # (batch, num_frames * num_patches, llm_dim)
        num_vision_tokens = vision_embeds.shape[1]
        
        # Ego state embedding - 1 token
        # ego_state shape: (batch, 1) for speed only, (batch, 3) for extended
        ego_input = ego_state.float()
        speed_embeds = self.speed_embedding(ego_input)  # (batch, llm_dim)
        speed_embeds = speed_embeds.unsqueeze(1)  # (batch, 1, llm_dim)
        
        # Navigation command embedding - 1 token (optional)
        if self.use_nav_cmd and nav_cmd_idx is not None:
            nav_embeds = self.nav_cmd_embedding(nav_cmd_idx)  # (batch, llm_dim)
            nav_embeds = nav_embeds.unsqueeze(1)  # (batch, 1, llm_dim)
        else:
            nav_embeds = None
        
        # Prepare text prompt (paper format from Figure 5); CoT always on: R2 then caption
        prompt = "USER: <image> List critical objects (e.g. lead vehicle, traffic lights) and describe the traffic scene. ASSISTANT: "
        prompt_inputs = self.tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True,
        ).to(device)
        prompt_embeds = self.language_model.get_input_embeddings()(prompt_inputs.input_ids).to(device)
        
        # Trajectory query tokens (paper: 10 learnable queries appended at end)
        traj_queries = self.trajectory_queries.expand(batch_size, -1, -1)  # (batch, 10, llm_dim)
        
        # Match dtypes and device (8-bit quantization can leave embeddings on CPU)
        vision_embeds = vision_embeds.to(device=device, dtype=prompt_embeds.dtype)
        traj_queries = traj_queries.to(device=device, dtype=prompt_embeds.dtype)
        speed_embeds = speed_embeds.to(device=device, dtype=prompt_embeds.dtype)
        if nav_embeds is not None:
            nav_embeds = nav_embeds.to(device=device, dtype=prompt_embeds.dtype)
        
        # During inference: generate captions if not provided
        # During training: captions should be GT captions (passed explicitly)
        if captions is None:
            with torch.no_grad():
                # Use all frames for caption generation
                captions = self.generate_caption(images, ego_state, nav_cmd_idx=nav_cmd_idx)
        
        caption_inputs = self.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        caption_embeds = self.language_model.get_input_embeddings()(caption_inputs.input_ids).to(device=device, dtype=prompt_embeds.dtype)
        
        # Sequence: [vision] + [speed] + [nav_cmd?] + [prompt] + [caption] + [traj_queries]
        if nav_embeds is not None:
            combined_embeds = torch.cat([
                vision_embeds, speed_embeds, nav_embeds, prompt_embeds, caption_embeds, traj_queries
            ], dim=1)
            attention_mask = torch.cat([
                torch.ones(batch_size, num_vision_tokens, device=device),
                torch.ones(batch_size, 1, device=device),  # speed
                torch.ones(batch_size, 1, device=device),  # nav_cmd
                prompt_inputs.attention_mask,
                caption_inputs.attention_mask,
                torch.ones(batch_size, self.num_trajectory_queries, device=device),
            ], dim=1)
            prefix_len = num_vision_tokens + 2 + prompt_inputs.input_ids.shape[1]  # +2 for speed + nav
        else:
            combined_embeds = torch.cat([
                vision_embeds, speed_embeds, prompt_embeds, caption_embeds, traj_queries
            ], dim=1)
            attention_mask = torch.cat([
                torch.ones(batch_size, num_vision_tokens, device=device),
                torch.ones(batch_size, 1, device=device),
                prompt_inputs.attention_mask,
                caption_inputs.attention_mask,
                torch.ones(batch_size, self.num_trajectory_queries, device=device),
            ], dim=1)
            prefix_len = num_vision_tokens + 1 + prompt_inputs.input_ids.shape[1]
        
        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get hidden states
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, llm_dim)
        
        # Extract trajectory query outputs (last 10 tokens as per Figure 5)
        traj_query_outputs = hidden_states[:, -self.num_trajectory_queries:, :]  # (batch, 10, llm_dim)
        
        traj_q = traj_query_outputs.float()
        trajectory_loss = None
        if self.use_diffusion_trajectory:
            cond = traj_q.mean(dim=1)  # (batch, llm_dim)
            result_diffusion = {'diffusion_cond': cond}
            if trajectories is not None:
                trajectory_loss, pred_trajectory, t_b, mse_per = (
                    self.traj_diffusion.training_loss_and_x0_hat(trajectories, cond)
                )
                result_diffusion['diffusion_t'] = t_b.detach()
                result_diffusion['diffusion_mse_per'] = mse_per
            else:
                pred_trajectory = self.traj_diffusion.sample(cond)
        else:
            pred_trajectory = self.trajectory_mlp(traj_q)
            if trajectories is not None:
                trajectory_loss = F.mse_loss(pred_trajectory, trajectories)
        
        result = {
            'pred_trajectory': pred_trajectory,
            'hidden_states': hidden_states,
            'traj_query_outputs': traj_query_outputs,
        }
        if self.use_diffusion_trajectory:
            result.update(result_diffusion)
        
        # Calculate losses if training (trajectories provided)
        if trajectories is not None:
            # Task 2: Trajectory — MSE (MLP) or noise prediction (diffusion)
            result['trajectory_loss'] = trajectory_loss
            
            # Smoothing loss: L2 penalty on acceleration (standard in trajectory prediction)
            pred_velocity = pred_trajectory[:, 1:, :] - pred_trajectory[:, :-1, :]  # (B, 9, 3)
            pred_accel = pred_velocity[:, 1:, :] - pred_velocity[:, :-1, :]  # (B, 8, 3)
            smoothing_loss = torch.mean(pred_accel ** 2)
            result['smoothing_loss'] = smoothing_loss
            
            # Task 1: Caption Generation — original full caption CE loss
            logits = outputs.logits
            caption_len = caption_inputs.input_ids.shape[1]
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            targets = caption_inputs.input_ids
            caption_logits = logits[:, prefix_len - 1 : prefix_len + caption_len - 1, :]  # (B, L, V)
            
            caption_loss = F.cross_entropy(
                caption_logits.reshape(-1, caption_logits.size(-1)),
                targets.reshape(-1),
                ignore_index=pad_id,
            )
            result['caption_loss'] = caption_loss
            
            # Extra R2 loss: CE on tokens before first \n (= R2 prefix boundary)
            # Caption is "Critical objects: ...\nScene description..."
            # Use [-1] because SentencePiece tokenizers prepend a space token when encoding "\n" standalone
            loss = self.config.caption_weight * caption_loss + self.config.trajectory_weight * trajectory_loss + self.smoothing_weight * smoothing_loss
            newline_id = self.tokenizer.encode("\n", add_special_tokens=False)[-1]
            n_r2 = (targets == newline_id).float().argmax(dim=1)  # (B,) first \n position per sample
            # Verify R2 decode on first call
            if not hasattr(self, '_r2_verified'):
                self._r2_verified = True
                B = targets.shape[0]
                print(f"\n  ── R2 loss verification (first batch, {B} samples) ──")
                print(f"  newline_id={newline_id}  encode('\\n')={self.tokenizer.encode(chr(10), add_special_tokens=False)}")
                print(f"  caption_len={caption_len}  pad_id={pad_id}")
                for b in range(min(B, 3)):
                    r2_decoded = self.tokenizer.decode(targets[b, :n_r2[b]])
                    rest_decoded = self.tokenizer.decode(targets[b, n_r2[b]:n_r2[b]+10])
                    print(f"  sample {b}: n_r2={n_r2[b].item():3d} | R2='{r2_decoded}'")
                    print(f"             rest='{rest_decoded}...'")
                print(f"  ──────────────────────────────────────────────\n")
            t_ar = torch.arange(caption_len, device=targets.device)
            r2_mask = (t_ar.unsqueeze(0) < n_r2.unsqueeze(1)) & (targets != pad_id)
            assert r2_mask.any(), "No R2 tokens found — no newline in tokenized caption"
            ce_none = F.cross_entropy(caption_logits.reshape(-1, caption_logits.size(-1)), targets.reshape(-1), ignore_index=pad_id, reduction="none").view(targets.shape)
            r2_loss = ce_none[r2_mask].mean()
            loss = loss + self.config.caption_r2_weight * r2_loss
            result['r2_loss'] = r2_loss
            result['loss'] = loss
        
        return result
    
    @torch.no_grad()
    def generate_caption(
        self,
        images: torch.Tensor,
        ego_state: torch.Tensor,
        nav_cmd_idx: torch.Tensor,
        max_length: int = 100,
    ) -> List[str]:
        """
        Generate driving scene captions conditioned on vision + ego state + nav command.
        
        Args:
            images: (B, num_frames, C, H, W) batch of multi-frame images
            ego_state: (B, D) where D=1 for speed only, D=3 for extended [vEgo, aEgo, steering]
            nav_cmd_idx: (B,) navigation command indices (0=LEFT, 1=RIGHT, 2=STRAIGHT)
        """
        self.eval()
        device = images.device
        batch_size = images.shape[0]
        dtype = next(self.language_model.parameters()).dtype
        
        # 1. Encode all frames with CLIP and project to LLM space
        vision_embeds = self.encode_temporal_images(images).to(device=device, dtype=dtype)
        
        # 2. Add ego state embedding
        speed_embeds = self.speed_embedding(ego_state.float().to(device))
        speed_embeds = speed_embeds.to(device=device, dtype=dtype).unsqueeze(1)
        
        # 3. Add nav command embedding (always required)
        nav_embeds = self.nav_cmd_embedding(nav_cmd_idx.to(device))
        nav_embeds = nav_embeds.to(device=device, dtype=dtype).unsqueeze(1)
        
        # 4. Use SAME prompt as training (must match forward()); CoT always on
        prompt = "USER: <image> List critical objects (e.g. lead vehicle, traffic lights) and describe the traffic scene. ASSISTANT: "
        prompt_inputs = self.tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True,
        ).to(device)
        prompt_embeds = self.language_model.get_input_embeddings()(prompt_inputs.input_ids).to(device=device, dtype=dtype)
        
        # 5. Combine: [Vision] + [Speed] + [Nav] + [Prompt] (matches training forward())
        combined_embeds = torch.cat([vision_embeds, speed_embeds, nav_embeds, prompt_embeds], dim=1)
        attention_mask = torch.ones(batch_size, combined_embeds.shape[1], device=device)
        
        # 5. Generate using HuggingFace generate() with inputs_embeds
        prefix_length = combined_embeds.shape[1]
        
        # Pass dummy input_ids on device so generate() keeps all internal tensors on CUDA
        # (8-bit quantized models can misplace tensors without this)
        dummy_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        outputs = self.language_model.generate(
            input_ids=dummy_input_ids,
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            do_sample=False,  # Greedy for consistency
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # DEBUG: Uncomment to debug token generation
        # print(f"DEBUG: prefix_length={prefix_length}, outputs.shape={outputs.shape}")
        # print(f"DEBUG: first 10 tokens: {outputs[0, :10].tolist()}")
        
        # 6. Decode tokens
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Clean up
        captions = [cap.strip() for cap in captions]
        
        return captions
    
    @torch.no_grad()
    def predict(
        self, 
        images: torch.Tensor, 
        ego_state: torch.Tensor,
        nav_cmd_idx,
        caption: str = None,
        caption_mode: str = "pred",
    ) -> Dict:
        """
        Make prediction for single sample with multi-frame input.
        
        Paper Table 4 shows two inference modes with different ADE results:
        - "Pred. caption" (caption_mode="pred"): ADE 0.955 - generate caption, then predict trajectory
        - "GT caption" (caption_mode="gt"): ADE 0.814 - use GT caption for trajectory (oracle)
        
        Args:
            images: (num_frames, C, H, W) or (1, num_frames, C, H, W) multi-frame input
            ego_state: (D,) ego state - D=1 for speed only, D=3 for extended [vEgo, aEgo, steering]
            nav_cmd_idx: Navigation command index (0=LEFT, 1=RIGHT, 2=STRAIGHT)
            caption: Ground truth caption (required if caption_mode="gt")
            caption_mode: How to use captions for trajectory prediction
                - "pred": Generate caption first, use it for trajectory (default)
                - "gt": Use provided GT caption for trajectory (oracle mode, better ADE)
        
        Returns:
            - trajectory: (10, 3) predicted waypoints
            - caption: The caption used for prediction
        """
        self.eval()
        
        # Handle input shape: (num_frames, C, H, W) -> (1, num_frames, C, H, W)
        if images.dim() == 4:
            images = images.unsqueeze(0)
        
        device = next(self.parameters()).device
        images = images.to(device)
        
        # Ensure ego_state is batched tensor
        if not isinstance(ego_state, torch.Tensor):
            ego_state = torch.tensor(ego_state, device=device)
        if ego_state.dim() == 1:
            ego_state = ego_state.unsqueeze(0)  # (D,) -> (1, D)
        ego_state = ego_state.to(device)
        
        # Ensure nav_cmd_idx is batched tensor
        if not isinstance(nav_cmd_idx, torch.Tensor):
            nav_cmd_idx = torch.tensor([nav_cmd_idx], dtype=torch.long, device=device)
        elif nav_cmd_idx.dim() == 0:
            nav_cmd_idx = nav_cmd_idx.unsqueeze(0)
        nav_cmd_idx = nav_cmd_idx.to(device)
        
        # Prepare caption based on mode
        if caption_mode == "gt":
            if caption is None:
                raise ValueError("caption_mode='gt' requires caption to be provided")
            caption_for_trajectory = caption
        else:
            # Pred caption mode: generate caption conditioned on all frames + ego_state + nav_cmd
            generated_captions = self.generate_caption(images, ego_state, nav_cmd_idx)
            caption_for_trajectory = generated_captions[0]
        
        # Get trajectory using the caption (either GT or predicted)
        output = self.forward(images, captions=[caption_for_trajectory], ego_state=ego_state, nav_cmd_idx=nav_cmd_idx)
        trajectory = output['pred_trajectory'][0].cpu().numpy()
        
        out = {
            'trajectory': trajectory,
            'caption': caption_for_trajectory,
        }
        if self.use_diffusion_trajectory and output.get('diffusion_cond') is not None:
            out['diffusion_cond'] = output['diffusion_cond']
        return out


# =============================================================================
# Metrics (from paper Section 4.2)
# =============================================================================

def compute_ade(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Average Displacement Error.
    
    Paper: "Mean Euclidean distance between predicted and ground truth 
    trajectory points over all time steps."
    """
    # pred, gt: (batch, num_points, 3) or (num_points, 3)
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    
    # Euclidean distance at each point
    distances = torch.sqrt(((pred - gt) ** 2).sum(dim=-1))  # (batch, num_points)
    
    # Mean over all points and batches
    ade = distances.mean().item()
    return ade


def compute_fde(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Final Displacement Error.
    
    Paper: "Euclidean distance between the predicted final point and 
    the ground truth final point."
    """
    # pred, gt: (batch, num_points, 3) or (num_points, 3)
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    
    # Distance at final point
    final_dist = torch.sqrt(((pred[:, -1, :] - gt[:, -1, :]) ** 2).sum(dim=-1))
    
    # Mean over batches
    fde = final_dist.mean().item()
    return fde


# =============================================================================
# Trainer (following paper's training procedure)
# =============================================================================

class CoVLATrainerPaper:
    """
    Trainer following paper's experiment setup.
    
    - Combined loss: 0.5 * caption_loss + 0.5 * trajectory_loss
    - Optional trajectory distillation: distill_traj_weight * MSE(student pred, frozen teacher pred);
      distill_traj_feat_weight * MSE(projector(student traj-query hiddens), teacher traj-query hiddens)
    - Metrics: ADE, FDE
    """
    
    def __init__(
        self,
        model: CoVLAAgentPaper,
        config: CoVLAConfig,
        teacher: Optional[CoVLAAgentPaper] = None,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        self.teacher = teacher
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
        
        # Optimizer
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
        self.scheduler = None  # Will be set in train()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.device == "cuda" else None
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_ade': [],
            'val_fde': [],
        }
    
    def _teacher_distill_forward(
        self,
        images: torch.Tensor,
        captions: List[str],
        ego_state: torch.Tensor,
        nav_cmd_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One frozen teacher forward (GT captions): (pred_trajectory, traj_query_outputs) on images.device."""
        t_dev = next(self.teacher.parameters()).device
        with torch.no_grad():
            te = self.teacher(
                images.to(t_dev),
                captions=captions,
                trajectories=None,
                ego_state=ego_state.to(t_dev),
                nav_cmd_idx=nav_cmd_idx.to(t_dev),
            )
        dev = images.device
        return te['pred_trajectory'].to(dev), te['traj_query_outputs'].to(dev)
    
    def _teacher_pred_trajectory(
        self,
        images: torch.Tensor,
        captions: List[str],
        ego_state: torch.Tensor,
        nav_cmd_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Frozen teacher trajectory prediction (viz / eval). Same forward as distillation."""
        return self._teacher_distill_forward(images, captions, ego_state, nav_cmd_idx)[0]
    
    def _compute_distill_losses(
        self,
        images: torch.Tensor,
        captions: List[str],
        ego_state: torch.Tensor,
        nav_cmd_idx: torch.Tensor,
        output: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-term MSE vs teacher; one teacher forward if any distill weight > 0."""
        device = images.device
        zero = torch.tensor(0.0, device=device)
        w_traj = self.config.distill_traj_weight
        w_feat = self.config.distill_traj_feat_weight
        if w_traj <= 0 and w_feat <= 0:
            return zero, zero
        t_pred, t_feat = self._teacher_distill_forward(images, captions, ego_state, nav_cmd_idx)
        distill_traj = (
            F.mse_loss(output['pred_trajectory'], t_pred.to(dtype=output['pred_trajectory'].dtype))
            if w_traj > 0
            else zero
        )
        distill_traj_feat = (
            F.mse_loss(
                self.model.traj_feat_projector(output['traj_query_outputs']).float(),
                t_feat.float(),
            )
            if w_feat > 0
            else zero
        )
        return distill_traj, distill_traj_feat
    
    def _distill_header_suffix(self) -> str:
        """Extra table columns when trajectory / feature distillation is enabled."""
        s = ""
        if self.config.distill_traj_weight > 0:
            s += f"{'dTraj':<10}"
        if self.config.distill_traj_feat_weight > 0:
            s += f"{'dFeat':<10}"
        return s
    
    def _distill_metrics_suffix(self, train_metrics: Dict[str, float]) -> str:
        s = ""
        if self.config.distill_traj_weight > 0:
            s += f"{train_metrics['distill_traj_loss']:<10.4f}"
        if self.config.distill_traj_feat_weight > 0:
            s += f"{train_metrics['distill_traj_feat_loss']:<10.4f}"
        return s
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Training uses GT captions for both:
        1. Caption loss (cross-entropy, train caption generation)
        2. Trajectory conditioning (MSE loss, train trajectory prediction)
        
        Single forward pass with GT captions - simple and stable.
        """
        self.model.train()
        
        total_loss = 0
        total_traj_loss = 0
        total_caption_loss = 0
        total_r2_loss = 0
        total_smooth_loss = 0
        total_distill_traj = 0
        total_distill_traj_feat = 0
        n_batches = 0
        global_step = getattr(self, '_global_step', -1)
        accum_steps = self.config.gradient_accumulation_steps
        num_bins = self.config.diffusion_loss_bins
        diff_bin_sum = [0.0] * num_bins
        diff_bin_cnt = [0] * num_bins
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(self.device)  # (B, num_frames, C, H, W)
            trajectories = batch['trajectory'].to(self.device)
            captions = batch['caption']  # GT captions for both conditioning and loss
            ego_state = batch['ego_state'].to(self.device)
            nav_cmd_idx = batch['nav_cmd_idx'].to(self.device)
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.model(
                        images,
                        captions=captions,
                        trajectories=trajectories,
                        ego_state=ego_state,
                        nav_cmd_idx=nav_cmd_idx,
                    )
                    distill_traj, distill_traj_feat = self._compute_distill_losses(
                        images, captions, ego_state, nav_cmd_idx, output
                    )
                    loss = (
                        output["loss"]
                        + self.config.distill_traj_weight * distill_traj
                        + self.config.distill_traj_feat_weight * distill_traj_feat
                    ) / accum_steps
                
                self.scaler.scale(loss).backward()
                
                # Step optimizer every accum_steps
                if (batch_idx + 1) % accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(
                    images,
                    captions=captions,
                    trajectories=trajectories,
                    ego_state=ego_state,
                    nav_cmd_idx=nav_cmd_idx,
                )
                distill_traj, distill_traj_feat = self._compute_distill_losses(
                    images, captions, ego_state, nav_cmd_idx, output
                )
                loss = (
                    output["loss"]
                    + self.config.distill_traj_weight * distill_traj
                    + self.config.distill_traj_feat_weight * distill_traj_feat
                ) / accum_steps
                loss.backward()
                
                if (batch_idx + 1) % accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * accum_steps  # Unscale for logging
            total_traj_loss += output.get('trajectory_loss', torch.tensor(0)).item()
            total_caption_loss += output.get('caption_loss', torch.tensor(0)).item()
            total_r2_loss += output['r2_loss'].item()
            total_smooth_loss += output.get('smoothing_loss', torch.tensor(0)).item()
            total_distill_traj += distill_traj.item()
            total_distill_traj_feat += distill_traj_feat.item()
            n_batches += 1
            global_step += 1
            
            if output.get('diffusion_t') is not None:
                t_b = output['diffusion_t']
                mse_b = output['diffusion_mse_per']
                Tn = self.model.traj_diffusion.num_timesteps
                bin_idx = (t_b.float() * num_bins / Tn).long().clamp(0, num_bins - 1)
                for bi in range(num_bins):
                    m = bin_idx == bi
                    if m.any():
                        diff_bin_sum[bi] += mse_b[m].sum().item()
                        diff_bin_cnt[bi] += int(m.sum().item())
            
            # Visualize every 500 steps
            if global_step % 250 == 0:
                self._visualize_training_sample(batch, output, global_step)
        
        self._global_step = global_step
        
        diffusion_bin_mse = [
            diff_bin_sum[i] / diff_bin_cnt[i] if diff_bin_cnt[i] else float('nan')
            for i in range(num_bins)
        ]
        
        return {
            'loss': total_loss / n_batches,
            'trajectory_loss': total_traj_loss / n_batches,
            'caption_loss': total_caption_loss / n_batches,
            'r2_loss': total_r2_loss / n_batches,
            'smoothing_loss': total_smooth_loss / n_batches,
            'distill_traj_loss': total_distill_traj / n_batches,
            'distill_traj_feat_loss': total_distill_traj_feat / n_batches,
            'diffusion_bin_mse': diffusion_bin_mse,
        }
    
    def _visualize_training_sample(self, batch, output, step):
        """Visualize a training sample (first item in batch)."""
        import matplotlib.pyplot as plt
        
        # Clear previous visualization only, then reprint epoch summaries
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
            
            # Reprint training info and epoch summaries
            if hasattr(self, '_training_info'):
                for line in self._training_info:
                    print(line)
            
            if hasattr(self, '_header'):
                print(self._header)
                print("-" * len(self._header))
            
            if hasattr(self, '_epoch_summaries') and self._epoch_summaries:
                for summary in self._epoch_summaries:
                    print(summary)
                print()
        except ImportError:
            pass
        
        self.model.eval()
        
        # Get first sample from batch
        gt_traj = batch['trajectory'][0].cpu().numpy()
        
        # Get nav command
        nav_cmd = batch['nav_cmd'][0]
        nav_cmd_idx = batch['nav_cmd_idx'][0:1].to(self.device)
        
        # Re-compute prediction in eval mode (training output has dropout noise)
        images_input = batch['images'][0:1].to(self.device)  # (1, num_frames, C, H, W)
        gt_tensor = batch['trajectory'][0:1].to(self.device)
        
        # With GT in forward: same as val — diffusion uses x0_hat (noise estimate), not full sample().
        # Without GT, diffusion would plot pure sample() which is often chaotic until sampling is well trained.
        with torch.no_grad():
            eval_output = self.model(
                images_input,
                captions=[batch['caption'][0]],
                trajectories=gt_tensor,
                ego_state=batch['ego_state'][0:1].to(self.device),
                nav_cmd_idx=nav_cmd_idx,
            )
        pred_traj = eval_output['pred_trajectory'][0].cpu().numpy()
        # Diffusion: optional extra full sample() curves (same cond); Pred(x0_hat) is always above
        sample_trajs = []
        if self.model.config.use_diffusion_trajectory:
            cmap = plt.cm.tab10
            dc = eval_output["diffusion_cond"]
            n_s = max(0, int(self.model.config.diffusion_viz_num_samples))
            for _ in range(n_s):
                sample_trajs.append(self.model.traj_diffusion.sample(dc).cpu().numpy()[0])
        else:
            cmap = None
        caption = batch['caption'][0] if isinstance(batch['caption'], list) else batch['caption']
        
        teacher_traj = None
        if self.teacher is not None:
            with torch.no_grad():
                tt = self._teacher_pred_trajectory(
                    images_input,
                    [caption],
                    batch['ego_state'][0:1].to(self.device),
                    nav_cmd_idx,
                )
            teacher_traj = tt[0].cpu().numpy()
        
        # Get ego state and denormalize
        ego_state_raw = batch['ego_state'][0].cpu().numpy()  # (3,) normalized
        ego = denormalize_ego_state(ego_state_raw)
        
        # Get matrices - DataLoader collates as [row][col][batch_idx]
        extrinsic = np.array([[col[0].item() for col in row] for row in batch['extrinsic_matrix']])
        intrinsic = np.array([[col[0].item() for col in row] for row in batch['intrinsic_matrix']])
        
        # Load frames from paths (original resolution for consistent display)
        # DataLoader collates lists as: batch['image_paths'][frame_idx][batch_idx]
        # So first sample's first frame = batch['image_paths'][0][0]
        # And first sample's last frame = batch['image_paths'][-1][0]
        first_path = batch['image_paths'][0][0]   # First frame of first sample
        last_path = batch['image_paths'][-1][0]   # Last frame of first sample
        first_frame = np.array(Image.open(first_path).convert('RGB'))
        last_frame = np.array(Image.open(last_path).convert('RGB'))
        
        # Metrics: val-aligned on x0_hat; optional sample ADE (first sample curve)
        ade = np.mean(np.linalg.norm(pred_traj - gt_traj, axis=1))
        fde = np.linalg.norm(pred_traj[-1] - gt_traj[-1])
        ade_s = (
            np.mean(np.linalg.norm(sample_trajs[0] - gt_traj, axis=1))
            if sample_trajs
            else None
        )
        
        # Generate predicted caption (uses all frames)
        with torch.no_grad():
            ego_tensor = batch['ego_state'][0:1].to(self.device)
            nav_cmd_idx = batch['nav_cmd_idx'][0:1].to(self.device)
            pred_caption = self.model.generate_caption(images_input, ego_tensor, nav_cmd_idx)[0]
        
        # Get physics trajectory (always available like gt_traj)
        traj_physics = batch['trajectory_physics'][0].cpu().numpy()
        
        # Create figure: 3 panels [First Frame | Current Frame | Bird's Eye View]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Left: First frame (t-1s)
        ax1 = axes[0]
        ax1.imshow(first_frame)
        ax1.set_title(f"First Frame (t-1s)")
        ax1.axis('off')
        
        # Middle: Current frame with trajectory
        ax2 = axes[1]
        plot_trajectory_on_image(last_frame, gt_traj, extrinsic, intrinsic, color='green', label='GT', ax=ax2, imshow_frame=True)
        pred_lbl = "Pred (x0_hat)" if self.model.config.use_diffusion_trajectory else "Pred"
        plot_trajectory_on_image(last_frame, pred_traj, extrinsic, intrinsic, color='red', label=pred_lbl, ax=ax2, imshow_frame=False)
        if sample_trajs:
            plot_trajectory_on_image(
                last_frame, sample_trajs[0], extrinsic, intrinsic,
                color='magenta', linestyle='--', label='Pred (sample)', ax=ax2, imshow_frame=False,
            )
            for j, st in enumerate(sample_trajs[1:], start=1):
                plot_trajectory_on_image(
                    last_frame, st, extrinsic, intrinsic,
                    color=cmap((j % 10) / 9.0), label=(f"sample+{j}" if j <= 2 else None),
                    ax=ax2, imshow_frame=False,
                )
        if teacher_traj is not None:
            plot_trajectory_on_image(
                last_frame, teacher_traj, extrinsic, intrinsic, color='darkorange', label='Teacher', ax=ax2, imshow_frame=False
            )
        plot_trajectory_on_image(last_frame, traj_physics, extrinsic, intrinsic, color='blue', label='Physics', ax=ax2, imshow_frame=False)
        ax2.legend(loc='upper right')
        samp_note = f" | ADE_s {ade_s:.2f}m" if ade_s is not None else ""
        ax2.set_title(f"Current Frame | Nav: {nav_cmd} | ADE {ade:.2f}m | FDE {fde:.2f}m | {ego['vEgo']:.1f} m/s{samp_note}")
        
        # Right: Bird's eye view
        ax3 = axes[2]
        ax3.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-o', markersize=5, label='GT')
        ax3.plot(pred_traj[:, 0], pred_traj[:, 1], 'r-o', markersize=5, label=pred_lbl)
        if teacher_traj is not None:
            ax3.plot(
                teacher_traj[:, 0],
                teacher_traj[:, 1],
                color='darkorange',
                marker='^',
                markersize=4,
                label='Teacher',
                linewidth=1.5,
            )
        ax3.plot(traj_physics[:, 0], traj_physics[:, 1], 'b-s', markersize=4, 
                 label='Physics', alpha=0.7, linewidth=1.5)
        if sample_trajs:
            ax3.plot(sample_trajs[0][:, 0], sample_trajs[0][:, 1], '--', color='magenta', linewidth=2, markersize=4, label='Pred (sample)')
            for j, st in enumerate(sample_trajs[1:], start=1):
                ax3.plot(st[:, 0], st[:, 1], '-', color=cmap((j % 10) / 9.0), linewidth=1.2, alpha=0.85)
        ax3.scatter([0], [0], c='blue', s=100, marker='*', label='Ego', zorder=5)
        ax3.set_xlabel('Forward (m)')
        ax3.set_ylabel('Lateral (m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        bev_note = f" | +{len(sample_trajs)} sample()" if sample_trajs else ""
        ax3.set_title(f"Bird's Eye View{bev_note}")
        
        plt.suptitle(f"Step {step}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Print captions
        print(f"📝 GT:   {caption[:500]}...")
        print(f"🤖 Pred: {pred_caption[:500]}...")
        
        self.model.train()
    
    @torch.no_grad()
    def evaluate(self, dataloader, compute_r2: bool = True) -> Dict[str, float]:
        """Evaluate on validation/test set. Accepts Dataset or DataLoader.
        When compute_r2 is False, skips caption generation (faster); R2 fields are NaN."""
        if isinstance(dataloader, Dataset):
            print(f"Evaluating {len(dataloader)} samples (batch_size={self.config.eval_batch_size})...")
            dataloader = DataLoader(dataloader, batch_size=self.config.eval_batch_size, shuffle=False)
        self.model.eval()
        
        total_loss = 0
        all_pred = []
        all_gt = []
        n_batches = 0
        # R2 (lead vehicle) metrics: precision & recall for "lead" class
        n_gt_lead = 0
        n_gt_none = 0
        n_pred_lead = 0
        n_correct_lead = 0
        n_correct_none = 0
        r2_dist_errors = []
        
        for batch in dataloader:
            images = batch['images'].to(self.device)  # (B, num_frames, C, H, W)
            trajectories = batch['trajectory'].to(self.device)
            captions = batch['caption']
            ego_state = batch['ego_state'].to(self.device)
            nav_cmd_idx = batch['nav_cmd_idx'].to(self.device)
            
            output = self.model(images, captions=captions, trajectories=trajectories,
                              ego_state=ego_state, nav_cmd_idx=nav_cmd_idx)
            
            total_loss += output['loss'].item()
            all_pred.append(output['pred_trajectory'].cpu())
            all_gt.append(trajectories.cpu())
            n_batches += 1
            
            # R2: generate captions and parse lead vehicle (GT vs pred) — optional (expensive)
            if compute_r2:
                pred_captions = self.model.generate_caption(images, ego_state, nav_cmd_idx)
                for gt_cap, pred_cap in zip(captions, pred_captions):
                    has_lead_gt, x_gt, y_gt = parse_r2_from_caption(gt_cap)
                    has_lead_pred, x_pred, y_pred = parse_r2_from_caption(pred_cap)
                    if has_lead_pred:
                        n_pred_lead += 1
                    if has_lead_gt:
                        n_gt_lead += 1
                        if has_lead_pred:
                            n_correct_lead += 1
                            if x_gt is not None and y_gt is not None and x_pred is not None and y_pred is not None:
                                dist = np.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
                                r2_dist_errors.append(float(dist))
                    else:
                        n_gt_none += 1
                        if not has_lead_pred:
                            n_correct_none += 1
        
        # Trajectory metrics
        all_pred = torch.cat(all_pred, dim=0)
        all_gt = torch.cat(all_gt, dim=0)
        ade = compute_ade(all_pred, all_gt)
        fde = compute_fde(all_pred, all_gt)
        
        # R2 metrics: precision = when we predicted lead, % correct; recall = when GT lead, % we predicted lead
        if not compute_r2:
            nan = float('nan')
            r2_acc = r2_recall = r2_precision = r2_ratio_lead = nan
            r2_dist_mean = nan
            n_gt_lead = n_correct_lead = 0
        else:
            total_r2 = n_gt_lead + n_gt_none
            n_correct_presence = n_correct_lead + n_correct_none
            r2_acc = (n_correct_presence / total_r2) if total_r2 else 0.0
            r2_recall = (n_correct_lead / n_gt_lead) if n_gt_lead else 0.0   # TP / (TP+FN)
            r2_precision = (n_correct_lead / n_pred_lead) if n_pred_lead else 0.0  # TP / (TP+FP)
            r2_ratio_lead = (n_gt_lead / total_r2) if total_r2 else 0.0      # fraction of val samples with lead
            r2_dist_mean = float(np.mean(r2_dist_errors)) if r2_dist_errors else float('nan')
        
        return {
            'loss': total_loss / n_batches,
            'ade': ade,
            'fde': fde,
            'r2_acc': r2_acc,
            'r2_recall': r2_recall,
            'r2_precision': r2_precision,
            'r2_ratio_lead': r2_ratio_lead,
            'r2_dist_mean': r2_dist_mean,
            'r2_n_lead': n_gt_lead,
            'r2_n_correct_lead': n_correct_lead,
        }
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        num_epochs: int = None,
    ):
        """Full training loop."""
        w_traj, w_feat = self.config.distill_traj_weight, self.config.distill_traj_feat_weight
        if w_traj > 0 or w_feat > 0:
            errs = []
            if self.teacher is None:
                errs.append("Pass teacher=CoVLAAgentPaper(...) to CoVLATrainerPaper.")
            if w_feat > 0 and self.config.distill_teacher_llm_dim is None:
                errs.append("distill_traj_feat_weight > 0 requires distill_teacher_llm_dim (teacher LM hidden size).")
            if w_feat > 0 and getattr(self.model, "traj_feat_projector", None) is None:
                errs.append("Feature distillation requires student built with distill_traj_feat_weight > 0 and distill_teacher_llm_dim set.")
            if errs:
                raise ValueError("Distillation setup: " + " ".join(errs))
        
        num_epochs = num_epochs or self.config.num_epochs
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
            )
        
        # Cosine annealing LR scheduler (decays LR from initial to 1/3 of initial)
        eta_min = self.config.learning_rate / 3
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=eta_min
        )
        
        dist_parts = []
        if self.config.distill_traj_weight > 0:
            dist_parts.append(f"traj pred w={self.config.distill_traj_weight}")
        if self.config.distill_traj_feat_weight > 0:
            dist_parts.append(
                f"traj-query hiddens w={self.config.distill_traj_feat_weight} "
                f"(teacher_dim={self.config.distill_teacher_llm_dim})"
            )
        dist_line = "Distillation: " + ", ".join(dist_parts) if dist_parts else None
        
        # Store training info for reprinting after clear_output
        self._training_info = [
            "",
            "=" * 70,
            "Training CoVLA-Agent (Paper Implementation)",
            "=" * 70,
            f"Train samples: {len(train_dataset)}",
            f"Val samples: {len(val_dataset)}" if val_dataset else None,
            f"Epochs: {num_epochs}",
            f"Batch size: {self.config.batch_size}",
            f"Learning rate: {self.config.learning_rate} → {self.config.learning_rate/3:.1e} (cosine decay)",
            f"Loss weights: caption={self.config.caption_weight}, trajectory={self.config.trajectory_weight}",
            dist_line,
            "=" * 70,
            "",
        ]
        self._training_info = [line for line in self._training_info if line is not None]
        
        for line in self._training_info:
            print(line)
        
        base = f"{'Epoch':<6}{'Loss':<10}{'Traj':<10}{'Cap':<10}{'R2':<10}{'Smooth':<10}"
        header = base + self._distill_header_suffix() + f"{'Val':<10}{'ADE':<8}{'FDE':<8}"
        self._header = header
        print(header)
        print("-" * len(header))
        
        best_ade = float('inf')
        self._epoch_summaries = []  # Store for reprinting after clear_output
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            
            # Validate (R2 via generate_caption from eval_r2_from_epoch onward, and always on last epoch)
            if val_loader:
                epoch_1based = epoch + 1
                compute_r2 = (
                    epoch_1based >= self.config.eval_r2_from_epoch
                    or epoch == num_epochs - 1
                )
                val_metrics = self.evaluate(val_loader, compute_r2=compute_r2)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_ade'].append(val_metrics['ade'])
                self.history['val_fde'].append(val_metrics['fde'])
                self.history.setdefault('r2_acc', []).append(val_metrics['r2_acc'])
                self.history.setdefault('r2_recall', []).append(val_metrics['r2_recall'])
                self.history.setdefault('r2_precision', []).append(val_metrics['r2_precision'])
                self.history.setdefault('r2_dist_mean', []).append(val_metrics['r2_dist_mean'])
                
                summary = (f"{epoch+1:<6}{train_metrics['loss']:<10.4f}"
                          f"{train_metrics['trajectory_loss']:<10.4f}"
                          f"{train_metrics['caption_loss']:<10.4f}"
                          f"{train_metrics['r2_loss']:<10.4f}"
                          f"{train_metrics['smoothing_loss']:<10.4f}"
                          f"{self._distill_metrics_suffix(train_metrics)}"
                          f"{val_metrics['loss']:<10.4f}"
                          f"{val_metrics['ade']:<8.3f}{val_metrics['fde']:<8.3f}")
                print(summary)
                self._epoch_summaries.append(summary)
                # R2 (lead vehicle): precision & recall for "lead" class, dist_err when both have coords
                if compute_r2:
                    r2_dist = val_metrics['r2_dist_mean']
                    r2_dist_s = f"{r2_dist:.2f}" if not np.isnan(r2_dist) else "n/a"
                    print(f"      R2: acc={val_metrics['r2_acc']:.3f} prec={val_metrics['r2_precision']:.3f} rec={val_metrics['r2_recall']:.3f} "
                          f"lead_ratio={val_metrics['r2_ratio_lead']:.2f} dist_err(m)={r2_dist_s} "
                          f"({val_metrics['r2_n_correct_lead']}/{val_metrics['r2_n_lead']} lead detected)")
                else:
                    print(f"      R2: (skipped — runs from epoch {self.config.eval_r2_from_epoch}+ and on final epoch)")
                if self.config.use_diffusion_trajectory:
                    bm = train_metrics.get('diffusion_bin_mse', [])
                    if bm:
                        parts = [f"{x:.4f}" if x == x else "nan" for x in bm]
                        print(f"      diff ε-MSE by t-bin (low t → high t): [{', '.join(parts)}]")
                
                # Save best model
                if val_metrics['ade'] < best_ade:
                    best_ade = val_metrics['ade']
                    self.model.save_trainable("covla_best.pt")
            else:
                summary = (f"{epoch+1:<6}{train_metrics['loss']:<10.4f}"
                          f"{train_metrics['trajectory_loss']:<10.4f}"
                          f"{train_metrics['caption_loss']:<10.4f}"
                          f"{train_metrics['r2_loss']:<10.4f}"
                          f"{train_metrics['smoothing_loss']:<10.4f}"
                          f"{self._distill_metrics_suffix(train_metrics)}")
                print(summary)
                self._epoch_summaries.append(summary)
                if self.config.use_diffusion_trajectory:
                    bm = train_metrics.get('diffusion_bin_mse', [])
                    if bm:
                        parts = [f"{x:.4f}" if x == x else "nan" for x in bm]
                        print(f"      diff ε-MSE by t-bin (low t → high t): [{', '.join(parts)}]")
            
            # Save checkpoint each epoch
            self.model.save_trainable(f"covla_epoch_{epoch+1}.pt")
            
            # Step LR scheduler
            self.scheduler.step()
        
        print("\n✓ Training complete!")
        
        # Print paper comparison
        if val_loader:
            final_ade = self.history['val_ade'][-1]
            final_fde = self.history['val_fde'][-1]
            print(f"\nFinal Results:")
            print(f"  ADE: {final_ade:.3f} (paper with predicted captions: 0.955)")
            print(f"  FDE: {final_fde:.3f} (paper with predicted captions: 2.239)")
            print(f"  Best ADE: {best_ade:.3f} (saved as covla_best.pt)")
        
        return self.history


def load_model(path: str = "covla_trainable.pt", device: str = "cuda") -> CoVLAAgentPaper:
    """
    Load a trained model (efficient - loads only trainable weights).
    
    Usage:
        model = load_model("covla_trainable.pt")
        result = model.predict(image, ego_state, nav_cmd_idx, caption_mode="pred")
    
    Continue training (use load_trainable so your config.learning_rate is used):
        config.learning_rate = 6.7e-6
        model = CoVLAAgentPaper(config); model.load_trainable("covla_best.pt")
        trainer = CoVLATrainerPaper(model, config); trainer.train(..., num_epochs=2)
    """
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get('config', CoVLAConfig(device=device))
    config.device = device
    
    model = CoVLAAgentPaper(config)
    model.load_trainable(path)
    model.eval()
    
    return model


# =============================================================================
# Visualization (following tutorial.ipynb style)
# =============================================================================

def device_to_camera(P_device: np.ndarray, extrinsic_matrix: np.ndarray) -> np.ndarray:
    """Convert device coordinates to camera coordinates."""
    P_device_hom = np.append(P_device, 1)
    P_camera_hom = np.dot(extrinsic_matrix, P_device_hom)
    return P_camera_hom[:3]


def camera_to_image(P_camera: np.ndarray, intrinsic_matrix: np.ndarray) -> np.ndarray:
    """Convert camera coordinates to image coordinates."""
    P_image_hom = np.dot(intrinsic_matrix, P_camera)
    P_image = P_image_hom[:2] / P_image_hom[2]
    return P_image


def plot_trajectory_on_image(
    frame: np.ndarray,
    trajectory: np.ndarray,
    extrinsic_matrix: np.ndarray,
    intrinsic_matrix: np.ndarray,
    color: str = "red",
    label: str = "Trajectory",
    ax=None,
    imshow_frame: bool = True,
    linestyle: str = "-",
):
    """
    Plot trajectory on image (following tutorial.ipynb style).
    
    Args:
        frame: Image array (H, W, 3)
        trajectory: Trajectory points (N, 3) in device coordinates
        extrinsic_matrix: Camera extrinsic (3, 4) or (4, 4)
        intrinsic_matrix: Camera intrinsic (3, 3)
        color: Trajectory color
        label: Legend label
        ax: Matplotlib axis (creates new if None)
        imshow_frame: If False, skip ax.imshow (overlay more curves on same axes)
        linestyle: Matplotlib line style for the polyline
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Ensure extrinsic is (3, 4)
    if extrinsic_matrix.shape[0] == 4:
        extrinsic_matrix = extrinsic_matrix[:3, :]
    
    # Convert trajectory to camera then image coordinates
    traj_camera = np.array([device_to_camera(p, extrinsic_matrix) for p in trajectory])
    
    # Keep only points in front of camera (z > 0)
    valid_mask = traj_camera[:, 2] > 0
    traj_camera = traj_camera[valid_mask]
    
    if len(traj_camera) == 0:
        if imshow_frame:
            ax.imshow(frame)
        ax.set_title("No trajectory points visible")
        ax.axis('off')
        return ax
    
    # Convert to image coordinates
    traj_image = np.array([camera_to_image(p, intrinsic_matrix) for p in traj_camera])
    
    # Filter points within image bounds
    h, w = frame.shape[:2]
    valid_mask = (
        (traj_image[:, 0] >= 0) & (traj_image[:, 0] < w) &
        (traj_image[:, 1] >= 0) & (traj_image[:, 1] < h)
    )
    traj_image = traj_image[valid_mask]
    
    # Plot
    if imshow_frame:
        ax.imshow(frame)
    if len(traj_image) > 0:
        ax.plot(traj_image[:, 0], traj_image[:, 1], 
               marker='o', color=color, linestyle=linestyle, 
               linewidth=3, markersize=8, alpha=0.9, label=label)
    ax.axis('off')
    
    return ax


def visualize_sample(
    sample_idx: int,
    image_files: List[str],
    states: List[Dict],
    captions_data: List[Dict],
    num_traj_points: int = 60,
):
    """
    Simple function to visualize a sample with trajectory overlay.
    
    Args:
        sample_idx: Index into the data arrays
        image_files: List of image file paths
        states: List of state dicts (with trajectory, extrinsic_matrix, etc.)
        captions_data: List of caption dicts
        num_traj_points: Number of trajectory points to show (default 60 = 3s at 20Hz)
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    if sample_idx >= len(image_files) or sample_idx >= len(states):
        print(f"Error: sample_idx {sample_idx} out of range")
        return
    
    # Load data
    sample_image = image_files[sample_idx]
    state = states[sample_idx]
    caption = captions_data[sample_idx] if sample_idx < len(captions_data) else {}
    
    # Load frame
    frame = np.array(Image.open(sample_image))
    
    # Extract trajectory and matrices
    trajectory = np.array(state['trajectory'][:num_traj_points])
    extrinsic = np.array(state['extrinsic_matrix'])
    intrinsic = np.array(state['intrinsic_matrix'])
    
    # Get speed (required field - called 'vEgo' in dataset)
    if 'vEgo' not in state:
        raise KeyError(f"'vEgo' not found in state. Available keys: {list(state.keys())}")
    speed = state['vEgo']
    aEgo = state.get('aEgo', None)
    steer = state.get('steeringAngleDeg', None)
    
    # Get caption
    caption_text = caption.get('rich_caption', caption.get('plain_caption', 'No caption'))
    
    # Get file info
    video_id = os.path.basename(os.path.dirname(sample_image))
    frame_name = os.path.basename(sample_image)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Image with trajectory
    plot_trajectory_on_image(frame, trajectory, extrinsic, intrinsic, 
                             color='lime', label='GT Trajectory', ax=axes[0])
    title = f"idx={sample_idx} | {video_id}/{frame_name} | {speed:.1f} m/s"
    if aEgo is not None and steer is not None:
        title += f" | Accel: {aEgo:.1f} | Steer: {steer:.0f}°"
    axes[0].set_title(title)
    axes[0].legend(loc='upper right')
    
    # Right: Bird's eye view
    axes[1].plot(trajectory[:, 0], trajectory[:, 1], 'g-o', markersize=3, label='Trajectory')
    axes[1].scatter([0], [0], c='red', s=100, marker='*', label='Ego', zorder=5)
    axes[1].set_xlabel('Forward (m)')
    axes[1].set_ylabel('Lateral (m)')
    axes[1].set_title(f"Bird's Eye View ({len(trajectory)} points)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Print info
    print(f"🚗 Speed: {speed:.1f} m/s", end="")
    if aEgo is not None and steer is not None:
        print(f" | Accel: {aEgo:.1f} m/s² | Steer: {steer:.0f}°")
    else:
        print()
    print(f"📍 First 5 trajectory points (x, y, z):")
    for i, pt in enumerate(trajectory[:5]):
        print(f"   [{i}]: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")
    print(f"📝 Caption:\n{caption_text[:500]}{'...' if len(caption_text) > 500 else ''}")
    
    return 


def visualize_dataset_sample(dataset, sample_idx: int):
    """
    Visualize a sample from CoVLADatasetPaper (train_dataset or val_dataset).
    Shows first frame (t-1s), current frame with trajectory, and bird's eye view.
    
    Usage:
        visualize_dataset_sample(val_dataset, 0)
        visualize_dataset_sample(train_dataset, 100)
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    if sample_idx >= len(dataset):
        print(f"Error: sample_idx {sample_idx} out of range (dataset has {len(dataset)} samples)")
        return
    
    sample = dataset[sample_idx]
    
    # Load frames:
    # - First frame: from tensor (224x224, no projection needed)
    # - Last frame: from path (original resolution for trajectory projection)
    first_frame, last_frame = _load_frames(sample)
    
    # Get data (handle both tensor and list/array formats)
    trajectory = sample['trajectory'].numpy() if hasattr(sample['trajectory'], 'numpy') else np.array(sample['trajectory'])
    extrinsic = sample['extrinsic_matrix'].numpy() if hasattr(sample['extrinsic_matrix'], 'numpy') else np.array(sample['extrinsic_matrix'])
    intrinsic = sample['intrinsic_matrix'].numpy() if hasattr(sample['intrinsic_matrix'], 'numpy') else np.array(sample['intrinsic_matrix'])
    ego_state_raw = sample['ego_state'].numpy() if hasattr(sample['ego_state'], 'numpy') else np.array(sample['ego_state'])
    caption = sample.get('caption', 'No caption')
    
    # Denormalize ego state
    ego = denormalize_ego_state(ego_state_raw)
    
    # Create figure - 3 panels
    num_frames = len(sample['image_paths'])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Left: First frame (history, t-1s)
    axes[0].imshow(first_frame)
    axes[0].set_title(f"First Frame (t-{num_frames-1}×0.5s)")
    axes[0].axis('off')
    
    # Middle: Current frame with trajectory
    plot_trajectory_on_image(last_frame, trajectory, extrinsic, intrinsic, 
                             color='lime', label='GT Trajectory', ax=axes[1])
    axes[1].set_title(f"Current Frame | {ego['vEgo']:.1f} m/s | Steer: {ego['steeringAngleDeg']:.0f}°")
    axes[1].legend(loc='upper right')
    
    # Right: Bird's eye view
    axes[2].plot(trajectory[:, 0], trajectory[:, 1], 'g-o', markersize=4, label='Trajectory')
    axes[2].scatter([0], [0], c='red', s=100, marker='*', label='Ego', zorder=5)
    axes[2].set_xlabel('Forward (m)')
    axes[2].set_ylabel('Lateral (m)')
    axes[2].set_title(f"Bird's Eye View ({len(trajectory)} points)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    
    plt.suptitle(f"Dataset sample {sample_idx}", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Print info
    print(f"🚗 Speed: {ego['vEgo']:.1f} m/s | Accel: {ego['aEgo']:.1f} m/s² | Steer: {ego['steeringAngleDeg']:.0f}°")
    print(f"📍 First 5 trajectory points (x, y, z):")
    for i, pt in enumerate(trajectory[:5]):
        print(f"   [{i}]: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")
    print(f"📝 Caption:\n{caption[:500]}{'...' if len(caption) > 500 else ''}")


def plot_training_curves(history: Dict):
    """Plot training curves."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], 'b-', label='Train')
    if history['val_loss']:
        axes[0].plot(history['val_loss'], 'r-', label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ADE
    if history['val_ade']:
        axes[1].plot(history['val_ade'], 'g-o')
        axes[1].axhline(y=0.955, color='r', linestyle='--', label='Paper (pred captions)')
        axes[1].axhline(y=0.814, color='b', linestyle='--', label='Paper (GT captions)')
        axes[1].set_title('Average Displacement Error')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('ADE (m)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # FDE
    if history['val_fde']:
        axes[2].plot(history['val_fde'], 'g-o')
        axes[2].axhline(y=2.239, color='r', linestyle='--', label='Paper (pred captions)')
        axes[2].axhline(y=1.655, color='b', linestyle='--', label='Paper (GT captions)')
        axes[2].set_title('Final Displacement Error')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('FDE (m)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║           CoVLA-Agent - Exact Paper Implementation                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Paper: https://arxiv.org/pdf/2408.10845                              ║
║                                                                       ║
║  Section 4: Experiments                                               ║
║  - Dataset: 70% train / 15% val / 15% test                            ║
║  - Frame sampling: 2Hz                                                ║
║  - Trajectory: 10 points (uniformly sampled from 60)                  ║
║  - Loss: 0.5 * caption_CE + 0.5 * trajectory_MSE                      ║
║  - Metrics: ADE, FDE                                                  ║
║                                                                       ║
║  Paper Results:                                                       ║
║  - Predicted captions: ADE=0.955, FDE=2.239                           ║
║  - GT captions: ADE=0.814, FDE=1.655                                  ║
╚═══════════════════════════════════════════════════════════════════════╝

USAGE IN COLAB:
===============

# 1. Load data (states, captions, image_files from previous cells)

# 2. Create datasets
config = CoVLAConfig(device="cuda")
train_dataset = CoVLADatasetPaper(states, captions_data, image_files, config, split="train")
val_dataset = CoVLADatasetPaper(states, captions_data, image_files, config, split="val")

# 3. Create model and trainer
model = CoVLAAgentPaper(config)
trainer = CoVLATrainerPaper(model, config)

# 4. Train
history = trainer.train(train_dataset, val_dataset, num_epochs=10)

# 5. Plot results
plot_training_curves(history)

# 6. Visualize predictions
visualize_prediction_paper(model, val_dataset[0])
""")


def _draw_trajectory(frame, trajectory, extrinsic, intrinsic, color):
    """Helper: project and draw trajectory on frame. Returns modified frame."""
    import cv2
    h, w = frame.shape[:2]
    if extrinsic.shape[0] == 4:
        extrinsic = extrinsic[:3, :]
    
    traj_cam = np.array([device_to_camera(p, extrinsic) for p in trajectory])
    valid = traj_cam[:, 2] > 0
    if not np.any(valid):
        return frame
    
    traj_img = np.array([camera_to_image(p, intrinsic) for p in traj_cam[valid]])
    mask = (traj_img[:, 0] >= 0) & (traj_img[:, 0] < w) & (traj_img[:, 1] >= 0) & (traj_img[:, 1] < h)
    traj_img = traj_img[mask]
    
    if len(traj_img) > 1:
        pts = traj_img.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, 3)
        for pt in traj_img:
            cv2.circle(frame, tuple(pt.astype(int)), 5, color, -1)
    return frame


def _to_numpy(x):
    """Convert tensor/list to numpy array."""
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.array(x)


def _load_frame(sample, debug=False):
    """Load high-res frame from sample (current frame) for projection."""
    path = sample['image_paths'][-1]
    return np.array(Image.open(path).convert('RGB'))


def _load_frames(sample, debug=False):
    """
    Load first and last frames from sample (both from paths for consistent resolution).
    """
    first_frame = np.array(Image.open(sample['image_paths'][0]).convert('RGB'))
    last_frame = np.array(Image.open(sample['image_paths'][-1]).convert('RGB'))
    return first_frame, last_frame


def _predict_sample(model, sample, caption_mode="pred", debug=False):
    """
    Run inference on a dataset sample. Returns dict with all needed data.
    
    Returns:
        dict with: first_frame, last_frame, gt_traj, pred_traj, trajectory_physics, extrinsic, intrinsic, 
                   ego_state, gt_caption, pred_caption, ade, fde, nav_cmd
    """
    device = next(model.parameters()).device
    
    # Extract data from sample
    first_frame, last_frame = _load_frames(sample, debug=debug)
    gt_traj = _to_numpy(sample['trajectory'])
    extrinsic = _to_numpy(sample['extrinsic_matrix'])
    intrinsic = _to_numpy(sample['intrinsic_matrix'])
    ego_state = sample['ego_state']  # (3,) tensor: [vEgo/30, aEgo/5, steering/500]
    gt_caption = sample.get('caption', '')
    nav_cmd = sample['nav_cmd']
    nav_cmd_idx = sample['nav_cmd_idx']
    
    # Physics-based trajectory (from IMU simulation)
    traj_physics = sample.get('trajectory_physics')
    if traj_physics is not None:
        traj_physics = _to_numpy(traj_physics)  # (10, 3)
    
    # Run inference with multi-frame input
    with torch.no_grad():
        images_input = sample['images'].unsqueeze(0).to(device)  # (1, num_frames, C, H, W)
        result = model.predict(
            images_input,
            ego_state,
            nav_cmd_idx,
            caption=gt_caption if caption_mode == "gt" else None,
            caption_mode=caption_mode,
        )
    pred_traj = result['trajectory']
    pred_caption = result.get('caption', gt_caption)
    
    # Compute metrics
    ade = compute_ade(torch.tensor(pred_traj).unsqueeze(0), torch.tensor(gt_traj).unsqueeze(0))
    fde = compute_fde(torch.tensor(pred_traj).unsqueeze(0), torch.tensor(gt_traj).unsqueeze(0))
    
    # Convert ego_state to numpy for display
    ego_state_np = ego_state.numpy() if hasattr(ego_state, 'numpy') else np.array(ego_state)
    
    return {
        'first_frame': first_frame,  # First frame (t-1s)
        'last_frame': last_frame,    # Current frame
        'gt_traj': gt_traj,
        'pred_traj': pred_traj,
        'trajectory_physics': traj_physics,  # (10, 3) or None - simulated from IMU
        'extrinsic': extrinsic,
        'intrinsic': intrinsic,
        'ego_state': ego_state_np,  # (3,) [vEgo/30, aEgo/5, steering/500] normalized
        'nav_cmd': nav_cmd,  # 'LEFT', 'RIGHT', or 'STRAIGHT'
        'gt_caption': gt_caption,
        'pred_caption': pred_caption,
        'ade': ade,
        'fde': fde,
        'diffusion_cond': result.get('diffusion_cond'),
    }


def visualize(
    model,
    dataset,
    idx: int = 0,
    caption_mode: str = "pred",
    diffusion_fan_k: Optional[int] = None,
):
    """
    Visualize model prediction on a dataset sample (matplotlib).
    
    With ``use_diffusion_trajectory``, optional K extra DDPM samples (fan) on camera + BEV.
    ``diffusion_fan_k=None`` uses ``config.diffusion_eval_num_samples``; set ``0`` to disable.
    
    Usage:
        visualize(model, val_dataset, idx=0)
        visualize(model, val_dataset, idx=10, caption_mode="gt")
        visualize(model, val_dataset, idx=0, diffusion_fan_k=8)
    """
    import matplotlib.pyplot as plt
    
    sample = dataset[idx]
    r = _predict_sample(model, sample, caption_mode)
    
    use_d = getattr(model.config, "use_diffusion_trajectory", False)
    if diffusion_fan_k is None:
        k_fan = getattr(model.config, "diffusion_eval_num_samples", 8) if use_d else 0
    else:
        k_fan = int(diffusion_fan_k) if use_d else 0
    
    fan_trajs = []
    if k_fan > 0 and r.get("diffusion_cond") is not None and getattr(model, "traj_diffusion", None) is not None:
        dc = r["diffusion_cond"]
        with torch.no_grad():
            for _ in range(k_fan):
                fan_trajs.append(model.traj_diffusion.sample(dc).cpu().numpy()[0])
    cmap = plt.cm.tab10
    
    # Get physics trajectory from prediction result
    traj_physics = r.get('trajectory_physics')  # (10, 3) or None
    
    # Create figure: 3 panels [First Frame | Current Frame | Bird's Eye View]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Denormalize ego state
    ego = denormalize_ego_state(r['ego_state'])
    nav_cmd = r['nav_cmd']
    
    # Left: First frame (t-1s)
    ax1 = axes[0]
    ax1.imshow(r['first_frame'])
    ax1.set_title(f"First Frame (t-1s)")
    ax1.axis('off')
    
    # Middle: Current frame with trajectories
    ax2 = axes[1]
    plot_trajectory_on_image(r['last_frame'], r['gt_traj'], r['extrinsic'], r['intrinsic'], 
                             color='green', label='GT', ax=ax2, imshow_frame=True)
    plot_trajectory_on_image(r['last_frame'], r['pred_traj'], r['extrinsic'], r['intrinsic'], 
                             color='red', label='Pred', ax=ax2, imshow_frame=False)
    
    # Plot physics trajectory on image if available
    if traj_physics is not None:
        plot_trajectory_on_image(r['last_frame'], traj_physics, r['extrinsic'], r['intrinsic'], 
                                 color='blue', label='Physics', ax=ax2, imshow_frame=False)
    
    for j, ft in enumerate(fan_trajs):
        plot_trajectory_on_image(
            r['last_frame'], ft, r['extrinsic'], r['intrinsic'],
            color=cmap((j % 10) / 9.0), label=(f"s{j}" if j < 4 else None),
            ax=ax2, imshow_frame=False,
        )
    
    ax2.legend(loc='upper right')
    fan_note = f" | fan K={len(fan_trajs)}" if fan_trajs else ""
    ax2.set_title(f"Current Frame | Nav: {nav_cmd} | ADE: {r['ade']:.2f}m | FDE: {r['fde']:.2f}m | {ego['vEgo']:.1f} m/s{fan_note}")
    
    # Right: Bird's eye view
    ax3 = axes[2]
    ax3.plot(r['gt_traj'][:, 0], r['gt_traj'][:, 1], 'g-o', markersize=5, label='GT')
    ax3.plot(r['pred_traj'][:, 0], r['pred_traj'][:, 1], 'r-o', markersize=5, label='Pred')
    
    # Plot physics trajectory in bird's eye view
    if traj_physics is not None:
        ax3.plot(traj_physics[:, 0], traj_physics[:, 1], 'b-s', markersize=4, 
                 label='Physics (IMU)', alpha=0.7, linewidth=1.5)
    
    for j, ft in enumerate(fan_trajs):
        ax3.plot(ft[:, 0], ft[:, 1], '-', color=cmap((j % 10) / 9.0), linewidth=1.2, alpha=0.85)
    
    ax3.scatter([0], [0], c='blue', s=100, marker='*', label='Ego', zorder=5)
    ax3.set_xlabel('Forward (m)')
    ax3.set_ylabel('Lateral (m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    bev_note = f" +{len(fan_trajs)} samples" if fan_trajs else ""
    ax3.set_title(f"Bird's Eye View{bev_note}")
    
    plt.tight_layout()
    plt.show()
    
    # Print ego state and nav cmd info
    print(f"🧭 Nav: {nav_cmd} | Speed: {ego['vEgo']:.1f} m/s | Accel: {ego['aEgo']:.1f} m/s² | Steer: {ego['steeringAngleDeg']:.0f}°")
    print(f"📝 GT Caption: {r['gt_caption'][:500]}...")
    print(f"📝 Pred Caption: {r['pred_caption'][:500]}...")
    if traj_physics is not None:
        print(f"📐 Physics trajectory available (10 points from IMU yaw rate)")


def generate_eval_images(
    model,
    dataset,
    output_dir: str = "eval",
    start_idx: int = 0,
    num_frames: int = 50,
    caption_mode: str = "pred",
    show_gt: bool = True,
    generate_video: bool = True,
    fps: int = 3,
    diffusion_fan_k: Optional[int] = None,
):
    """
    Generate evaluation images with trajectory overlay + bird's eye view + captions.
    Optionally generates a video from the images.
    
    Usage:
        generate_eval_images(model, val_dataset, "eval", num_frames=30)
        generate_eval_images(model, val_dataset, "eval", num_frames=30, generate_video=True, fps=5)
    
    Output: eval/0000.png, eval/0001.png, ... and eval/eval.mp4 (if generate_video=True)
    """
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    end_idx = min(start_idx + num_frames, len(dataset))
    print(f"🖼️ Generating {end_idx - start_idx} images → {output_dir}/")
    model.eval()
    
    use_d = getattr(model.config, "use_diffusion_trajectory", False)
    if diffusion_fan_k is None:
        k_fan = getattr(model.config, "diffusion_eval_num_samples", 8) if use_d else 0
    else:
        k_fan = int(diffusion_fan_k) if use_d else 0
    
    cmap = plt.cm.tab10
    metrics = []
    saved_images = []
    
    for i in tqdm(range(start_idx, end_idx), desc="Processing", mininterval=60.0):
        sample = dataset[i]
        r = _predict_sample(model, sample, caption_mode)
        metrics.append({'ade': r['ade'], 'fde': r['fde']})
        
        fan_trajs = []
        if k_fan > 0 and r.get("diffusion_cond") is not None and getattr(model, "traj_diffusion", None) is not None:
            dc = r["diffusion_cond"]
            with torch.no_grad():
                for _ in range(k_fan):
                    fan_trajs.append(model.traj_diffusion.sample(dc).cpu().numpy()[0])
        
        # Get GT caption from sample
        gt_caption = sample.get('caption', 'N/A')
        
        # Denormalize ego state and get nav_cmd
        ego = denormalize_ego_state(r['ego_state'])
        nav_cmd = r['nav_cmd']
        
        # Create figure with 3 columns: [First Frame | Current Frame | Bird's Eye View]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Left: First frame (t-1s)
        ax1 = axes[0]
        ax1.imshow(r['first_frame'])
        ax1.set_title(f"First Frame (t-1s)")
        ax1.axis('off')
        
        # Middle: Current frame with trajectory overlay
        ax2 = axes[1]
        plot_trajectory_on_image(r['last_frame'], r['gt_traj'], r['extrinsic'], r['intrinsic'], 
                                 color='green', label='GT', ax=ax2, imshow_frame=True)
        plot_trajectory_on_image(r['last_frame'], r['pred_traj'], r['extrinsic'], r['intrinsic'], 
                                 color='red', label='Pred', ax=ax2, imshow_frame=False)
        for j, ft in enumerate(fan_trajs):
            plot_trajectory_on_image(
                r['last_frame'], ft, r['extrinsic'], r['intrinsic'],
                color=cmap((j % 10) / 9.0), label=(f"s{j}" if j < 4 else None),
                ax=ax2, imshow_frame=False,
            )
        ax2.legend(loc='upper right')
        fan_note = f" | fan K={len(fan_trajs)}" if fan_trajs else ""
        ax2.set_title(f"Frame {i} | Nav: {nav_cmd} | ADE: {r['ade']:.2f}m | FDE: {r['fde']:.2f}m | {ego['vEgo']:.1f} m/s{fan_note}")
        
        # Right: Bird's eye view
        ax3 = axes[2]
        ax3.plot(r['gt_traj'][:, 0], r['gt_traj'][:, 1], 'g-o', markersize=6, linewidth=2, label='GT')
        ax3.plot(r['pred_traj'][:, 0], r['pred_traj'][:, 1], 'r-o', markersize=6, linewidth=2, label='Pred')
        for j, ft in enumerate(fan_trajs):
            ax3.plot(ft[:, 0], ft[:, 1], '-', color=cmap((j % 10) / 9.0), linewidth=1.2, alpha=0.85)
        ax3.scatter([0], [0], c='blue', s=150, marker='*', label='Ego', zorder=5)
        ax3.set_xlabel('Forward (m)')
        ax3.set_ylabel('Lateral (m)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        bev_note = f" +{len(fan_trajs)} samples" if fan_trajs else ""
        ax3.set_title(f"Bird's Eye View{bev_note}")
        
        plt.tight_layout()
        
        # Add captions as text below the figure
        gt_text = f"GT: {gt_caption[:500]}..." if len(gt_caption) > 500 else f"GT: {gt_caption}"
        pred_text = f"Pred: {r['pred_caption'][:500]}..." if len(r['pred_caption']) > 500 else f"Pred: {r['pred_caption']}"
        
        # Adjust layout first to make room for captions (bottom) and title (top)
        plt.subplots_adjust(bottom=0.22, top=0.92)
        
        # Place captions in the margin area (y in figure coordinates, 0=bottom, 1=top)
        fig.text(0.02, 0.12, gt_text, fontsize=9, color='green', wrap=True)
        fig.text(0.02, 0.02, pred_text, fontsize=9, color='red', wrap=True)
        
        # Save (no bbox_inches='tight' to ensure consistent image sizes for video)
        img_path = f"{output_dir}/{i-start_idx:04d}.png"
        plt.savefig(img_path, dpi=120)
        plt.close(fig)
        saved_images.append(img_path)
    
    avg_ade = np.mean([m['ade'] for m in metrics])
    avg_fde = np.mean([m['fde'] for m in metrics])
    print(f"✅ Saved {len(metrics)} images to {output_dir}/ | ADE: {avg_ade:.3f}m, FDE: {avg_fde:.3f}m")
    
    # Generate video from saved images
    if generate_video and saved_images:
        try:
            import imageio
            video_path = f"{output_dir}/eval.mp4"
            
            # Read images and write video
            writer = imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8)
            for img_path in saved_images:
                frame = imageio.imread(img_path)
                writer.append_data(frame)
            writer.close()
            
            print(f"🎬 Generated video: {video_path} ({len(saved_images)} frames @ {fps} fps)")
        except ImportError:
            print("⚠️ imageio not installed. Run: pip install imageio imageio-ffmpeg")
        except Exception as e:
            print(f"⚠️ Video generation failed: {e}")
    return {'output_dir': output_dir, 'num_frames': len(metrics), 'avg_ade': avg_ade, 'avg_fde': avg_fde}


def find_worst_predictions(
    model,
    dataset,
    output_dir: str = "worst_predictions",
    num_worst: int = 200,
    caption_mode: str = "pred",
    batch_size: int = 16,
):
    """
    Find and save the worst ADE predictions for error analysis.
    Uses batched inference for speed.
    
    Usage:
        results = find_worst_predictions(model, val_dataset, num_worst=200)
        results = find_worst_predictions(model, val_dataset, num_worst=200, batch_size=32)
    
    Output files: worst_predictions/0000_ade5.23_idx1234.png (ranked by ADE, worst first)
    """
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    
    # Create dataloader for batched inference
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Evaluate all samples in batches
    print(f"Evaluating {len(dataset)} samples (batch_size={batch_size})...")
    all_ade_fde = []  # Store (idx, ade, fde) for all samples
    
    sample_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing ADE", mininterval=120.0):
            images = batch['images'].to(device)  # (B, num_frames, C, H, W)
            trajectories = batch['trajectory'].to(device)
            captions = batch['caption'] if caption_mode == "gt" else None
            ego_state = batch['ego_state'].to(device)
            nav_cmd_idx = batch['nav_cmd_idx'].to(device)
            
            # Forward pass
            output = model(images, captions=captions, ego_state=ego_state, nav_cmd_idx=nav_cmd_idx)
            
            pred_trajs = output['pred_trajectory'].cpu()  # (B, 10, 3)
            gt_trajs = trajectories.cpu()  # (B, 10, 3)
            
            # Compute ADE/FDE for each sample in batch
            for i in range(len(images)):
                ade = compute_ade(pred_trajs[i:i+1], gt_trajs[i:i+1])
                fde = compute_fde(pred_trajs[i:i+1], gt_trajs[i:i+1])
                all_ade_fde.append({
                    'idx': sample_idx,
                    'ade': ade,
                    'fde': fde,
                })
                sample_idx += 1
    
    # Sort by ADE (worst first) and get top N indices
    all_ade_fde.sort(key=lambda x: x['ade'], reverse=True)
    
    actual_num_worst = min(num_worst, len(all_ade_fde))
    print(f"\nWorst {actual_num_worst} ADE range: {all_ade_fde[actual_num_worst-1]['ade']:.2f} - {all_ade_fde[0]['ade']:.2f} m")
    
    # Load, predict, and save worst samples in one pass
    print(f"Saving worst {actual_num_worst} predictions to {output_dir}/...")
    worst = []
    for rank, item in enumerate(tqdm(all_ade_fde[:actual_num_worst], desc="Processing worst")):
        idx = item['idx']
        sample = dataset[idx]
        r = _predict_sample(model, sample, caption_mode)
        
        worst.append({
            'idx': idx,
            'ade': item['ade'],
            'fde': item['fde'],
        })
        
        # Save visualization: 3 panels [First Frame | Current Frame | Bird's Eye View]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        ego = denormalize_ego_state(r['ego_state'])
        nav_cmd = r['nav_cmd']
        traj_physics = r.get('trajectory_physics')
        
        # Left: First frame (t-1s)
        ax1 = axes[0]
        ax1.imshow(r['first_frame'])
        ax1.set_title(f"First Frame (t-1s)")
        ax1.axis('off')
        
        # Middle: Current frame with trajectory
        ax2 = axes[1]
        plot_trajectory_on_image(r['last_frame'], r['gt_traj'], r['extrinsic'], r['intrinsic'], 
                                 color='green', label='GT', ax=ax2)
        plot_trajectory_on_image(r['last_frame'], r['pred_traj'], r['extrinsic'], r['intrinsic'], 
                                 color='red', label='Pred', ax=ax2)
        
        # Plot physics trajectory if available
        if traj_physics is not None:
            plot_trajectory_on_image(r['last_frame'], traj_physics, r['extrinsic'], r['intrinsic'], 
                                     color='blue', label='Physics', ax=ax2)
        
        ax2.legend(loc='upper right')
        ax2.set_title(f"Rank {rank+1} | Idx {item['idx']} | Nav: {nav_cmd} | ADE: {item['ade']:.2f}m | FDE: {item['fde']:.2f}m\n"
                      f"Speed: {ego['vEgo']:.1f} m/s | Steer: {ego['steeringAngleDeg']:.0f}°")
        
        # Right: Bird's eye view
        ax3 = axes[2]
        ax3.plot(r['gt_traj'][:, 0], r['gt_traj'][:, 1], 'g-o', markersize=6, linewidth=2, label='GT')
        ax3.plot(r['pred_traj'][:, 0], r['pred_traj'][:, 1], 'r-o', markersize=6, linewidth=2, label='Pred')
        
        # Plot physics trajectory in bird's eye view
        if traj_physics is not None:
            ax3.plot(traj_physics[:, 0], traj_physics[:, 1], 'b-s', markersize=4, 
                     linewidth=1.5, label='Physics', alpha=0.7)
        
        ax3.scatter([0], [0], c='blue', s=150, marker='*', label='Ego', zorder=5)
        ax3.set_xlabel('Forward (m)')
        ax3.set_ylabel('Lateral (m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        ax3.set_title("Bird's Eye View")
        
        plt.tight_layout()
        
        # Add captions (same layout as generate_eval_images)
        gt_caption = sample.get('caption', 'N/A')[:500]
        pred_caption = r['pred_caption'][:500]
        gt_text = f"GT: {gt_caption}..." if len(sample.get('caption', '')) > 500 else f"GT: {gt_caption}"
        pred_text = f"Pred: {pred_caption}..." if len(r['pred_caption']) > 500 else f"Pred: {pred_caption}"
        
        # Adjust layout to make room for captions (bottom) and title (top)
        plt.subplots_adjust(bottom=0.22, top=0.92)
        
        # Place captions in the margin area
        fig.text(0.02, 0.12, gt_text, fontsize=9, color='green', wrap=True)
        fig.text(0.02, 0.02, pred_text, fontsize=9, color='red', wrap=True)
        
        plt.savefig(f"{output_dir}/{rank:04d}_ade{item['ade']:.2f}_idx{item['idx']}.png", dpi=100)
        plt.close(fig)
    
    print(f"✅ Saved {len(worst)} worst predictions to {output_dir}/")
    
    # Return summary
    return {
        'worst': worst,  # List of {idx, ade, fde} for worst samples
        'all_ade_fde': all_ade_fde,  # List of {idx, ade, fde} for all samples
        'mean_ade': np.mean([r['ade'] for r in all_ade_fde]),
        'worst_ade': worst[0]['ade'] if worst else 0,
        'output_dir': output_dir,
    }

