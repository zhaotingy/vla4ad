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
    
    Set use_paper_model=True for exact paper architecture.
    Set use_paper_model=False for lightweight version (free Colab).
    """
    
    # Model selection
    use_paper_model: bool = False  # True = paper models, False = lightweight
    
    # Paper models (CLIP ViT-L + 7B LLM, ~24GB VRAM)
    # Using Mistral-7B as open alternative (no approval needed, similar quality)
    # Other options if you have access:
    # - "meta-llama/Llama-2-7b-hf" (requires Meta approval)
    # - "meta-llama/Llama-2-7b-chat-hf" (requires Meta approval)
    vision_encoder_paper: str = "openai/clip-vit-large-patch14"
    language_model_paper: str = "mistralai/Mistral-7B-Instruct-v0.2"  # Open, no approval
    
    # Lightweight models (CLIP ViT-B + TinyLlama, ~8GB VRAM)
    vision_encoder_light: str = "openai/clip-vit-base-patch32"
    language_model_light: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
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
    
    # Training
    batch_size: int = 8
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
    
    @property
    def vision_encoder(self) -> str:
        return self.vision_encoder_paper if self.use_paper_model else self.vision_encoder_light
    
    @property
    def language_model(self) -> str:
        return self.language_model_paper if self.use_paper_model else self.language_model_light


# =============================================================================
# Dataset (matching paper's preprocessing)
# =============================================================================

class CoVLADatasetPaper(Dataset):
    """
    Dataset following paper's Section 4.1 preprocessing:
    - Frames sampled at 2Hz
    - 10 trajectory points (uniformly sampled from 60)
    - Excludes frames without complete 3-second trajectory
    """
    
    def __init__(
        self,
        states_data: List[Dict],
        captions_data: List[Dict],
        image_files: List[str],
        config: CoVLAConfig,
        split: str = "train",  # "train", "val", or "test"
    ):
        self.config = config
        self.split = split
        
        # Sample at 2Hz (every 10th frame from 20Hz data)
        sample_interval = 20 // config.frame_sample_rate  # = 10
        
        # Filter and prepare samples with logging
        self.samples = []
        self.physics_mismatch_indices = []  # Store indices for debugging
        filter_counts = {
            'total_candidates': 0,
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
            caption_text = caption.get('rich_caption', caption.get('plain_caption', ''))
            nav_cmd = get_nav_cmd(caption_text)
            
            filter_counts['passed'] += 1
            self.samples.append({
                'image_path': image_files[i],
                'trajectory': sampled_trajectory,
                'trajectory_physics': trajectory_physics,  # Simulated from IMU (10 points or None)
                'caption': caption_text,
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
        print(f"   ├─ Incomplete trajectory: {filter_counts['incomplete_trajectory']}")
        print(f"   ├─ Absolute value >200m: {filter_counts['absolute_too_large']}")
        print(f"   ├─ Delta >20m: {filter_counts['delta_too_large']}")
        print(f"   ├─ Lateral delta >5m: {filter_counts['lateral_delta_too_large']}")
        print(f"   ├─ Physics mismatch: {filter_counts['physics_mismatch']}")
        print(f"   └─ No IMU data (kept): {filter_counts['no_imu_data']}")
        print(f"   ✓ Passed: {passed} ({100*passed/total:.1f}%)")
        print(f"   ✗ Filtered: {filtered} ({100*filtered/total:.1f}%)")
        
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
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, self.config.image_size, self.config.image_size)
        
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
            'image': image,
            'trajectory': trajectory,
            'trajectory_physics': traj_physics,  # (10, 3) or None - simulated from IMU
            'caption': sample['caption'],
            'ego_state': ego_state,  # (3,) [vEgo/30, aEgo/5, steering/500] normalized
            'nav_cmd': nav_cmd,  # string: 'LEFT', 'RIGHT', 'STRAIGHT'
            'nav_cmd_idx': nav_cmd_idx,  # tensor: 0, 1, or 2
            'extrinsic_matrix': sample.get('extrinsic_matrix'),
            'intrinsic_matrix': sample.get('intrinsic_matrix'),
            'image_path': sample['image_path'],
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
        
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.language_model,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        )
        
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
        
        # Trajectory MLP (paper specification)
        self.trajectory_mlp = TrajectoryMLP(
            input_dim=self.llm_dim,
            num_points=config.trajectory_points,
            coord_dim=config.trajectory_dim,
        )
        
        # Loss weights
        self.smoothing_weight = config.smoothing_weight
        
        # Print model info
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_type = "PAPER" if config.use_paper_model else "LIGHTWEIGHT"
        print(f"✓ CoVLA-Agent initialized ({model_type})")
        print(f"  Vision: {config.vision_encoder}")
        print(f"  Language: {config.language_model}")
        print(f"  Ego state: {'extended (vEgo, aEgo, steering)' if config.use_extended_ego_state else 'speed only'}")
        print(f"  Nav command: {'enabled (LEFT/RIGHT/STRAIGHT)' if config.use_nav_cmd else 'disabled'}")
        print(f"  Total params: {total:,}")
        print(f"  Trainable: {trainable:,}")
    
    def _apply_lora(self):
        """Apply LoRA to language model."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_rank * 2,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                task_type=TaskType.CAUSAL_LM,
            )
            self.language_model = get_peft_model(self.language_model, lora_config)
            print("✓ LoRA applied")
        except ImportError:
            print("⚠ PEFT not installed, training full model")
    
    def save_trainable(self, path: str = "covla_trainable.pt"):
        """
        Save only trainable components (efficient - ~50MB instead of ~5GB).
        
        Saves: LoRA adapters, vision_projection, speed_embedding, 
               trajectory_queries, trajectory_mlp
        """
        trainable_state = {
            'vision_projection': self.vision_projection.state_dict(),
            'speed_embedding': self.speed_embedding.state_dict(),
            'trajectory_queries': self.trajectory_queries.data,
            'trajectory_mlp': self.trajectory_mlp.state_dict(),
            'config': self.config,
        }
        
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
        self.trajectory_mlp.load_state_dict(checkpoint['trajectory_mlp'])
        
        # Load nav_cmd_embedding if present
        if 'nav_cmd_embedding' in checkpoint and self.use_nav_cmd:
            self.nav_cmd_embedding.load_state_dict(checkpoint['nav_cmd_embedding'])
        
        # Load LoRA weights
        if 'lora' in checkpoint and hasattr(self.language_model, 'peft_config'):
            current_state = self.language_model.state_dict()
            current_state.update(checkpoint['lora'])
            self.language_model.load_state_dict(current_state)
        
        # Move entire model to device
        self.to(self.config.device)
        
        print(f"✓ Loaded trainable weights from: {path}")
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using vision encoder.
        Returns all patch tokens (not just CLS) for richer visual representation.
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
            images: (batch, 3, H, W) input images
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
        
        # Encode images - get patch tokens (not just CLS)
        vision_features = self.encode_image(images)  # (batch, num_patches+1, vision_dim)
        vision_embeds = self.vision_projection(vision_features)  # (batch, num_patches+1, llm_dim)
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
        
        # Prepare text prompt (paper format from Figure 5)
        prompt = "USER: <image> Describe the traffic scene. ASSISTANT: "
        prompt_inputs = self.tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True,
        ).to(device)
        prompt_embeds = self.language_model.get_input_embeddings()(prompt_inputs.input_ids)
        
        # Trajectory query tokens (paper: 10 learnable queries appended at end)
        traj_queries = self.trajectory_queries.expand(batch_size, -1, -1)  # (batch, 10, llm_dim)
        
        # Match dtypes
        vision_embeds = vision_embeds.to(prompt_embeds.dtype)
        traj_queries = traj_queries.to(prompt_embeds.dtype)
        speed_embeds = speed_embeds.to(prompt_embeds.dtype)
        if nav_embeds is not None:
            nav_embeds = nav_embeds.to(prompt_embeds.dtype)
        
        # During inference: generate captions if not provided
        # During training: captions should be GT captions (passed explicitly)
        if captions is None:
            with torch.no_grad():
                captions = self.generate_caption(images, ego_state)
        
        caption_inputs = self.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        caption_embeds = self.language_model.get_input_embeddings()(caption_inputs.input_ids)
        caption_embeds = caption_embeds.to(prompt_embeds.dtype)
        
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
        
        # Predict trajectory from each query token (paper: MLP on trajectory queries)
        pred_trajectory = self.trajectory_mlp(traj_query_outputs.float())  # (batch, 10, 3)
        
        result = {
            'pred_trajectory': pred_trajectory,
            'hidden_states': hidden_states,
        }
        
        # Calculate losses if training (trajectories provided)
        if trajectories is not None:
            # Task 2: Trajectory Prediction (MSE loss)
            trajectory_loss = F.mse_loss(pred_trajectory, trajectories)
            result['trajectory_loss'] = trajectory_loss
            
            # Smoothing loss: L2 penalty on acceleration (standard in trajectory prediction)
            pred_velocity = pred_trajectory[:, 1:, :] - pred_trajectory[:, :-1, :]  # (B, 9, 3)
            pred_accel = pred_velocity[:, 1:, :] - pred_velocity[:, :-1, :]  # (B, 8, 3)
            smoothing_loss = torch.mean(pred_accel ** 2)
            result['smoothing_loss'] = smoothing_loss
            
            # Task 1: Caption Generation (Cross-Entropy loss)
            # Use outputs from same forward pass (captions are GT during training)
            logits = outputs.logits
            caption_len = caption_inputs.input_ids.shape[1]
            
            # Extract logits for caption positions (shifted by 1 for autoregressive prediction)
            caption_logits = logits[:, prefix_len-1:prefix_len+caption_len-1, :]
            
            caption_loss = F.cross_entropy(
                caption_logits.reshape(-1, caption_logits.size(-1)),
                caption_inputs.input_ids.reshape(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )
            result['caption_loss'] = caption_loss
            
            # Combined loss (paper: equally weighted + smoothing)
            result['loss'] = (
                self.config.caption_weight * caption_loss +
                self.config.trajectory_weight * trajectory_loss +
                self.smoothing_weight * smoothing_loss
            )
        
        return result
    
    @torch.no_grad()
    def generate_caption(
        self,
        images: torch.Tensor,
        ego_state: torch.Tensor,
        max_length: int = 100,
    ) -> List[str]:
        """
        Generate driving scene captions conditioned on vision + ego state.
        
        Args:
            images: (B, C, H, W) batch of images
            ego_state: (B, D) where D=1 for speed only, D=3 for extended [vEgo, aEgo, steering]
        """
        self.eval()
        device = images.device
        batch_size = images.shape[0]
        dtype = next(self.language_model.parameters()).dtype
        
        # 1. Encode image with CLIP and project to LLM space
        vision_features = self.encode_image(images)  # (B, num_patches, vision_dim)
        vision_embeds = self.vision_projection(vision_features).to(dtype)  # (B, num_patches, llm_dim)
        
        # 2. Add ego state embedding
        speed_embeds = self.speed_embedding(ego_state.float().to(device))  # (B, llm_dim)
        speed_embeds = speed_embeds.to(dtype).unsqueeze(1)  # (B, 1, llm_dim)
        
        # 3. Use SAME prompt as training (must match forward())
        prompt = "USER: <image> Describe the traffic scene. ASSISTANT: "
        prompt_inputs = self.tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True,
        ).to(device)
        prompt_embeds = self.language_model.get_input_embeddings()(prompt_inputs.input_ids).to(dtype)
        
        # 4. Combine: [Vision] + [Speed] + [Prompt] (matches training)
        combined_embeds = torch.cat([vision_embeds, speed_embeds, prompt_embeds], dim=1)
        attention_mask = torch.ones(batch_size, combined_embeds.shape[1], device=device)
        
        # 5. Generate using HuggingFace generate() with inputs_embeds
        prefix_length = combined_embeds.shape[1]
        
        outputs = self.language_model.generate(
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
        image: torch.Tensor, 
        ego_state: torch.Tensor,
        caption: str = None,
        caption_mode: str = "pred",
        nav_cmd_idx: torch.Tensor = None,  # Optional: navigation command index
    ) -> Dict:
        """
        Make prediction for single image.
        
        Paper Table 4 shows two inference modes with different ADE results:
        - "Pred. caption" (caption_mode="pred"): ADE 0.955 - generate caption, then predict trajectory
        - "GT caption" (caption_mode="gt"): ADE 0.814 - use GT caption for trajectory (oracle)
        
        Args:
            image: Input image tensor
            ego_state: (D,) ego state - D=1 for speed only, D=3 for extended [vEgo, aEgo, steering]
            caption: Ground truth caption (required if caption_mode="gt")
            caption_mode: How to use captions for trajectory prediction
                - "pred": Generate caption first, use it for trajectory (default)
                - "gt": Use provided GT caption for trajectory (oracle mode, better ADE)
            nav_cmd_idx: Navigation command index (0=LEFT, 1=RIGHT, 2=STRAIGHT)
        
        Returns:
            - trajectory: (10, 3) predicted waypoints
            - caption: The caption used for prediction
        """
        self.eval()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        device = next(self.parameters()).device
        image = image.to(device)
        
        # Ensure ego_state is batched tensor
        if not isinstance(ego_state, torch.Tensor):
            ego_state = torch.tensor(ego_state, device=device)
        if ego_state.dim() == 1:
            ego_state = ego_state.unsqueeze(0)  # (D,) -> (1, D)
        ego_state = ego_state.to(device)
        
        # Ensure nav_cmd_idx is batched tensor if provided
        if nav_cmd_idx is not None:
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
            # Pred caption mode: generate caption conditioned on image + ego_state
            generated_captions = self.generate_caption(image, ego_state)
            caption_for_trajectory = generated_captions[0]
        
        # Get trajectory using the caption (either GT or predicted)
        output = self.forward(image, captions=[caption_for_trajectory], ego_state=ego_state, nav_cmd_idx=nav_cmd_idx)
        trajectory = output['pred_trajectory'][0].cpu().numpy()
        
        return {
            'trajectory': trajectory,
            'caption': caption_for_trajectory,
        }


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
    - Metrics: ADE, FDE
    """
    
    def __init__(self, model: CoVLAAgentPaper, config: CoVLAConfig):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
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
        total_smooth_loss = 0
        n_batches = 0
        global_step = getattr(self, '_global_step', -1)
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            trajectories = batch['trajectory'].to(self.device)
            captions = batch['caption']  # GT captions for both conditioning and loss
            ego_state = batch['ego_state'].to(self.device)
            nav_cmd_idx = batch.get('nav_cmd_idx')
            if nav_cmd_idx is not None:
                nav_cmd_idx = nav_cmd_idx.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Single forward pass: GT captions condition trajectory AND compute caption loss
            if self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.model(
                        images, 
                        captions=captions,          # GT captions for trajectory conditioning
                        trajectories=trajectories,
                        ego_state=ego_state,
                        nav_cmd_idx=nav_cmd_idx,
                    )
                    loss = output['loss']
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(
                    images, 
                    captions=captions,
                    trajectories=trajectories,
                    ego_state=ego_state,
                    nav_cmd_idx=nav_cmd_idx,
                )
                loss = output['loss']
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            total_traj_loss += output.get('trajectory_loss', torch.tensor(0)).item()
            total_caption_loss += output.get('caption_loss', torch.tensor(0)).item()
            total_smooth_loss += output.get('smoothing_loss', torch.tensor(0)).item()
            n_batches += 1
            global_step += 1
            
            # Visualize every 500 steps (change to smaller number for debugging)
            if global_step % 500 == 0:
                self._visualize_training_sample(batch, output, global_step)
        
        self._global_step = global_step
        
        return {
            'loss': total_loss / n_batches,
            'trajectory_loss': total_traj_loss / n_batches,
            'caption_loss': total_caption_loss / n_batches,
            'smoothing_loss': total_smooth_loss / n_batches,
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
        
        # Re-compute prediction in eval mode (training output has dropout noise)
        with torch.no_grad():
            eval_output = self.model(
                batch['image'][0:1].to(self.device),
                captions=[batch['caption'][0]],
                ego_state=batch['ego_state'][0:1].to(self.device),
            )
        pred_traj = eval_output['pred_trajectory'][0].cpu().numpy()
        caption = batch['caption'][0] if isinstance(batch['caption'], list) else batch['caption']
        
        # Get ego state and denormalize
        ego_state_raw = batch['ego_state'][0].cpu().numpy()  # (3,) normalized
        ego = denormalize_ego_state(ego_state_raw)
        
        # Get nav command
        nav_cmd = batch.get('nav_cmd', ['STRAIGHT'])[0] if 'nav_cmd' in batch else 'N/A'
        
        # Get matrices - DataLoader collates as [row][col][batch_idx]
        extrinsic = np.array([[col[0].item() for col in row] for row in batch['extrinsic_matrix']])
        intrinsic = np.array([[col[0].item() for col in row] for row in batch['intrinsic_matrix']])
        
        # Load original image from path
        image_path = batch['image_path'][0] if isinstance(batch['image_path'], list) else batch['image_path']
        if os.path.exists(image_path):
            frame = np.array(Image.open(image_path).convert('RGB'))
        else:
            image = batch['image'][0]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            frame = ((image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Compute metrics
        ade = np.mean(np.linalg.norm(pred_traj - gt_traj, axis=1))
        fde = np.linalg.norm(pred_traj[-1] - gt_traj[-1])
        
        # Generate predicted caption
        with torch.no_grad():
            image_tensor = batch['image'][0:1].to(self.device)
            ego_tensor = batch['ego_state'][0:1].to(self.device)
            pred_caption = self.model.generate_caption(image_tensor, ego_tensor)[0]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Image with trajectory
        ax1 = axes[0]
        plot_trajectory_on_image(frame, gt_traj, extrinsic, intrinsic, color='green', label='GT', ax=ax1)
        plot_trajectory_on_image(frame, pred_traj, extrinsic, intrinsic, color='red', label='Pred', ax=ax1)
        ax1.legend(loc='upper right')
        # Title with ego state and nav_cmd info
        ax1.set_title(f"Step {step} | Nav: {nav_cmd} | ADE: {ade:.2f}m | FDE: {fde:.2f}m | {ego['vEgo']:.1f} m/s")
        
        # Right: Bird's eye view
        ax2 = axes[1]
        ax2.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-o', markersize=5, label='GT')
        ax2.plot(pred_traj[:, 0], pred_traj[:, 1], 'r-o', markersize=5, label='Pred')
        ax2.scatter([0], [0], c='blue', s=100, marker='*', label='Ego', zorder=5)
        ax2.set_xlabel('Forward (m)')
        ax2.set_ylabel('Lateral (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.set_title("Bird's Eye View")
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Print captions
        print(f"📝 GT:   {caption[:500]}...")
        print(f"🤖 Pred: {pred_caption[:500]}...")
        
        self.model.train()
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test set."""
        self.model.eval()
        
        total_loss = 0
        all_pred = []
        all_gt = []
        n_batches = 0
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            trajectories = batch['trajectory'].to(self.device)
            captions = batch['caption']
            ego_state = batch['ego_state'].to(self.device)
            nav_cmd_idx = batch.get('nav_cmd_idx')
            if nav_cmd_idx is not None:
                nav_cmd_idx = nav_cmd_idx.to(self.device)
            
            output = self.model(images, captions=captions, trajectories=trajectories, 
                              ego_state=ego_state, nav_cmd_idx=nav_cmd_idx)
            
            total_loss += output['loss'].item()
            all_pred.append(output['pred_trajectory'].cpu())
            all_gt.append(trajectories.cpu())
            n_batches += 1
        
        # Compute metrics
        all_pred = torch.cat(all_pred, dim=0)
        all_gt = torch.cat(all_gt, dim=0)
        
        ade = compute_ade(all_pred, all_gt)
        fde = compute_fde(all_pred, all_gt)
        
        return {
            'loss': total_loss / n_batches,
            'ade': ade,
            'fde': fde,
        }
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        num_epochs: int = None,
    ):
        """Full training loop."""
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
                batch_size=self.config.batch_size,
                shuffle=False,
            )
        
        # Cosine annealing LR scheduler (decays LR from initial to 1/3 of initial)
        eta_min = self.config.learning_rate / 3
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=eta_min
        )
        
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
            "=" * 70,
            "",
        ]
        self._training_info = [line for line in self._training_info if line is not None]
        
        for line in self._training_info:
            print(line)
        
        header = f"{'Epoch':<6}{'Loss':<10}{'Traj':<10}{'Cap':<10}{'Smooth':<10}{'Val':<10}{'ADE':<8}{'FDE':<8}"
        self._header = header
        print(header)
        print("-" * len(header))
        
        best_ade = float('inf')
        self._epoch_summaries = []  # Store for reprinting after clear_output
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_ade'].append(val_metrics['ade'])
                self.history['val_fde'].append(val_metrics['fde'])
                
                summary = (f"{epoch+1:<6}{train_metrics['loss']:<10.4f}"
                          f"{train_metrics['trajectory_loss']:<10.4f}"
                          f"{train_metrics['caption_loss']:<10.4f}"
                          f"{train_metrics['smoothing_loss']:<10.4f}"
                          f"{val_metrics['loss']:<10.4f}"
                          f"{val_metrics['ade']:<8.3f}{val_metrics['fde']:<8.3f}")
                print(summary)
                self._epoch_summaries.append(summary)
                
                # Save best model
                if val_metrics['ade'] < best_ade:
                    best_ade = val_metrics['ade']
                    self.model.save_trainable("covla_best.pt")
            else:
                summary = (f"{epoch+1:<6}{train_metrics['loss']:<10.4f}"
                          f"{train_metrics['trajectory_loss']:<10.4f}"
                          f"{train_metrics['caption_loss']:<10.4f}"
                          f"{train_metrics['smoothing_loss']:<10.4f}")
                print(summary)
                self._epoch_summaries.append(summary)
            
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
        result = model.predict(image, ego_state=ego_state, caption_mode="pred")
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
    ax.imshow(frame)
    if len(traj_image) > 0:
        ax.plot(traj_image[:, 0], traj_image[:, 1], 
               marker='o', color=color, linestyle='-', 
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
    
    # Load original image (not the resized tensor)
    image_path = sample.get('image_path', None)
    if image_path and os.path.exists(image_path):
        frame = np.array(Image.open(image_path))
    else:
        # Fallback: convert tensor to numpy (but this is resized)
        img_tensor = sample['image']
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame = ((img_tensor.cpu() * std + mean).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        print("⚠️ Using resized image (no image_path)")
    
    # Get data (handle both tensor and list/array formats)
    trajectory = sample['trajectory'].numpy() if hasattr(sample['trajectory'], 'numpy') else np.array(sample['trajectory'])
    extrinsic = sample['extrinsic_matrix'].numpy() if hasattr(sample['extrinsic_matrix'], 'numpy') else np.array(sample['extrinsic_matrix'])
    intrinsic = sample['intrinsic_matrix'].numpy() if hasattr(sample['intrinsic_matrix'], 'numpy') else np.array(sample['intrinsic_matrix'])
    ego_state_raw = sample['ego_state'].numpy() if hasattr(sample['ego_state'], 'numpy') else np.array(sample['ego_state'])
    caption = sample.get('caption', 'No caption')
    
    # Denormalize ego state
    ego = denormalize_ego_state(ego_state_raw)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Image with trajectory
    plot_trajectory_on_image(frame, trajectory, extrinsic, intrinsic, 
                             color='lime', label='GT Trajectory', ax=axes[0])
    axes[0].set_title(f"Dataset sample {sample_idx} | {ego['vEgo']:.1f} m/s | Accel: {ego['aEgo']:.1f} | Steer: {ego['steeringAngleDeg']:.0f}°")
    axes[0].legend(loc='upper right')
    
    # Right: Bird's eye view
    axes[1].plot(trajectory[:, 0], trajectory[:, 1], 'g-o', markersize=4, label='Trajectory')
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
    print(f"🚗 Speed: {ego['vEgo']:.1f} m/s | Accel: {ego['aEgo']:.1f} m/s² | Steer: {ego['steeringAngleDeg']:.0f}°")
    print(f"📍 First 5 trajectory points (x, y, z):")
    for i, pt in enumerate(trajectory[:5]):
        print(f"   [{i}]: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")
    print(f"📝 Caption:\n{caption[:500]}{'...' if len(caption) > 500 else ''}")


def visualize_inference(
    model,
    image: torch.Tensor,
    gt_trajectory: np.ndarray,
    extrinsic_matrix: np.ndarray,
    intrinsic_matrix: np.ndarray,
    ego_state: torch.Tensor,
    gt_caption: str = None,
    image_path: str = None,
    caption_mode: str = "pred",
):
    """
    Visualize model inference with trajectory overlay on image.
    
    For simpler API, use: visualize(model, dataset, idx)
    """
    import matplotlib.pyplot as plt
    
    result = model.predict(image, ego_state=ego_state, caption=gt_caption, caption_mode=caption_mode)
    pred_traj = result['trajectory']
    pred_caption = result['caption']
    
    # Load frame
    if image_path and os.path.exists(image_path):
        frame = np.array(Image.open(image_path).convert('RGB'))
    else:
        if image.dim() == 4:
            image = image[0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame = ((image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Metrics
    ade = compute_ade(torch.tensor(pred_traj).unsqueeze(0), torch.tensor(gt_trajectory).unsqueeze(0))
    fde = compute_fde(torch.tensor(pred_traj).unsqueeze(0), torch.tensor(gt_trajectory).unsqueeze(0))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_trajectory_on_image(frame, gt_trajectory, extrinsic_matrix, intrinsic_matrix, 
                             color='green', label='GT', ax=axes[0])
    plot_trajectory_on_image(frame, pred_traj, extrinsic_matrix, intrinsic_matrix, 
                             color='red', label='Pred', ax=axes[0])
    axes[0].legend()
    axes[0].set_title(f"ADE: {ade:.2f}m | FDE: {fde:.2f}m")
    
    axes[1].plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'g-o', markersize=5, label='GT')
    axes[1].plot(pred_traj[:, 0], pred_traj[:, 1], 'r-o', markersize=5, label='Pred')
    axes[1].scatter([0], [0], c='blue', s=100, marker='*', label='Ego')
    axes[1].set_xlabel('Forward (m)')
    axes[1].set_ylabel('Lateral (m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    axes[1].set_title("Bird's Eye View")
    
    plt.tight_layout()
    plt.show()
    
    print(f"📝 Caption: {pred_caption[:200]}...")
    return {'pred_trajectory': pred_traj, 'pred_caption': pred_caption, 'ade': ade, 'fde': fde, 'caption': pred_caption}


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
    """Load high-res frame from sample, fallback to tensor denormalization."""
    if 'image_path' in sample and os.path.exists(sample['image_path']):
        frame = np.array(Image.open(sample['image_path']).convert('RGB'))
        if debug:
            print(f"  Loaded from path: {sample['image_path']}, shape={frame.shape}, dtype={frame.dtype}")
        return frame
    # Denormalize tensor
    if debug:
        print(f"  No image_path, using tensor. Keys: {list(sample.keys())}")
    t = sample['image']
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    frame = (img * 255).astype(np.uint8)
    if debug:
        print(f"  Tensor denorm: shape={frame.shape}, min={frame.min()}, max={frame.max()}")
    return frame


def _predict_sample(model, sample, caption_mode="pred", debug=False):
    """
    Run inference on a dataset sample. Returns dict with all needed data.
    
    Returns:
        dict with: frame, gt_traj, pred_traj, trajectory_physics, extrinsic, intrinsic, 
                   ego_state, gt_caption, pred_caption, ade, fde, nav_cmd
    """
    device = next(model.parameters()).device
    
    # Extract data from sample
    frame = _load_frame(sample, debug=debug)
    gt_traj = _to_numpy(sample['trajectory'])
    extrinsic = _to_numpy(sample['extrinsic_matrix'])
    intrinsic = _to_numpy(sample['intrinsic_matrix'])
    ego_state = sample['ego_state']  # (3,) tensor: [vEgo/30, aEgo/5, steering/500]
    gt_caption = sample.get('caption', '')
    nav_cmd = sample.get('nav_cmd', 'STRAIGHT')
    nav_cmd_idx = sample.get('nav_cmd_idx', None)  # tensor or None
    
    # Physics-based trajectory (from IMU simulation)
    traj_physics = sample.get('trajectory_physics')
    if traj_physics is not None:
        traj_physics = _to_numpy(traj_physics)  # (10, 3)
    
    # Run inference
    with torch.no_grad():
        result = model.predict(
            sample['image'].unsqueeze(0).to(device),
            ego_state=ego_state,
            caption=gt_caption if caption_mode == "gt" else None,
            caption_mode=caption_mode,
            nav_cmd_idx=nav_cmd_idx,  # Pass nav command if available
        )
    pred_traj = result['trajectory']
    pred_caption = result.get('caption', gt_caption)
    
    # Compute metrics
    ade = compute_ade(torch.tensor(pred_traj).unsqueeze(0), torch.tensor(gt_traj).unsqueeze(0))
    fde = compute_fde(torch.tensor(pred_traj).unsqueeze(0), torch.tensor(gt_traj).unsqueeze(0))
    
    # Convert ego_state to numpy for display
    ego_state_np = ego_state.numpy() if hasattr(ego_state, 'numpy') else np.array(ego_state)
    
    return {
        'frame': frame,
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
    }


def visualize(model, dataset, idx: int = 0, caption_mode: str = "pred"):
    """
    Visualize model prediction on a dataset sample (matplotlib).
    
    Usage:
        visualize(model, val_dataset, idx=0)
        visualize(model, val_dataset, idx=10, caption_mode="gt")
    """
    import matplotlib.pyplot as plt
    
    sample = dataset[idx]
    r = _predict_sample(model, sample, caption_mode)
    
    # Get physics trajectory from prediction result
    traj_physics = r.get('trajectory_physics')  # (10, 3) or None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Image with trajectories
    ax1 = axes[0]
    plot_trajectory_on_image(r['frame'], r['gt_traj'], r['extrinsic'], r['intrinsic'], 
                             color='green', label='GT', ax=ax1)
    plot_trajectory_on_image(r['frame'], r['pred_traj'], r['extrinsic'], r['intrinsic'], 
                             color='red', label='Pred', ax=ax1)
    
    # Plot physics trajectory on image if available
    if traj_physics is not None:
        plot_trajectory_on_image(r['frame'], traj_physics, r['extrinsic'], r['intrinsic'], 
                                 color='blue', label='Physics', ax=ax1)
    
    ax1.legend(loc='upper right')
    
    # Denormalize ego state
    ego = denormalize_ego_state(r['ego_state'])
    nav_cmd = r.get('nav_cmd', 'N/A')
    
    # Title with nav_cmd and metrics
    ax1.set_title(f"Nav: {nav_cmd} | ADE: {r['ade']:.2f}m | FDE: {r['fde']:.2f}m | {ego['vEgo']:.1f} m/s")
    
    # Right: Bird's eye view
    ax2 = axes[1]
    ax2.plot(r['gt_traj'][:, 0], r['gt_traj'][:, 1], 'g-o', markersize=5, label='GT')
    ax2.plot(r['pred_traj'][:, 0], r['pred_traj'][:, 1], 'r-o', markersize=5, label='Pred')
    
    # Plot physics trajectory in bird's eye view
    if traj_physics is not None:
        ax2.plot(traj_physics[:, 0], traj_physics[:, 1], 'b-s', markersize=4, 
                 label='Physics (IMU)', alpha=0.7, linewidth=1.5)
    
    ax2.scatter([0], [0], c='blue', s=100, marker='*', label='Ego', zorder=5)
    ax2.set_xlabel('Forward (m)')
    ax2.set_ylabel('Lateral (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title("Bird's Eye View")
    
    plt.tight_layout()
    plt.show()
    
    # Print ego state and nav cmd info
    print(f"🧭 Nav: {nav_cmd} | Speed: {ego['vEgo']:.1f} m/s | Accel: {ego['aEgo']:.1f} m/s² | Steer: {ego['steeringAngleDeg']:.0f}°")
    print(f"📝 Caption: {r['pred_caption'][:500]}...")
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
    
    metrics = []
    saved_images = []
    
    for i in tqdm(range(start_idx, end_idx), desc="Processing", mininterval=5.0):
        sample = dataset[i]
        r = _predict_sample(model, sample, caption_mode)
        metrics.append({'ade': r['ade'], 'fde': r['fde']})
        
        # Get GT caption from sample
        gt_caption = sample.get('caption', 'N/A')
        
        # Create figure with 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Image with trajectory overlay
        ax1 = axes[0]
        plot_trajectory_on_image(r['frame'], r['gt_traj'], r['extrinsic'], r['intrinsic'], 
                                 color='green', label='GT', ax=ax1)
        plot_trajectory_on_image(r['frame'], r['pred_traj'], r['extrinsic'], r['intrinsic'], 
                                 color='red', label='Pred', ax=ax1)
        ax1.legend(loc='upper right')
        
        # Denormalize ego state and get nav_cmd
        ego = denormalize_ego_state(r['ego_state'])
        nav_cmd = r.get('nav_cmd', 'N/A')
        ax1.set_title(f"Frame {i} | Nav: {nav_cmd} | ADE: {r['ade']:.2f}m | FDE: {r['fde']:.2f}m | {ego['vEgo']:.1f} m/s")
        
        # Right: Bird's eye view
        ax2 = axes[1]
        ax2.plot(r['gt_traj'][:, 0], r['gt_traj'][:, 1], 'g-o', markersize=6, linewidth=2, label='GT')
        ax2.plot(r['pred_traj'][:, 0], r['pred_traj'][:, 1], 'r-o', markersize=6, linewidth=2, label='Pred')
        ax2.scatter([0], [0], c='blue', s=150, marker='*', label='Ego', zorder=5)
        ax2.set_xlabel('Forward (m)')
        ax2.set_ylabel('Lateral (m)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.set_title("Bird's Eye View")
        
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
            images = batch['image'].to(device)
            trajectories = batch['trajectory'].to(device)
            captions = batch['caption'] if caption_mode == "gt" else None
            ego_state = batch['ego_state'].to(device)
            
            # Forward pass
            output = model(images, captions=captions, ego_state=ego_state)
            
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
        
        # Save visualization immediately
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Image with trajectory
        ax1 = axes[0]
        plot_trajectory_on_image(r['frame'], r['gt_traj'], r['extrinsic'], r['intrinsic'], 
                                 color='green', label='GT', ax=ax1)
        plot_trajectory_on_image(r['frame'], r['pred_traj'], r['extrinsic'], r['intrinsic'], 
                                 color='red', label='Pred', ax=ax1)
        
        # Plot physics trajectory if available
        traj_physics = r.get('trajectory_physics')
        if traj_physics is not None:
            plot_trajectory_on_image(r['frame'], traj_physics, r['extrinsic'], r['intrinsic'], 
                                     color='blue', label='Physics', ax=ax1)
        
        ax1.legend(loc='upper right')
        
        ego = denormalize_ego_state(r['ego_state'])
        nav_cmd = r.get('nav_cmd', 'N/A')
        ax1.set_title(f"Rank {rank+1} | Idx {item['idx']} | Nav: {nav_cmd} | ADE: {item['ade']:.2f}m | FDE: {item['fde']:.2f}m\n"
                      f"Speed: {ego['vEgo']:.1f} m/s | Steer: {ego['steeringAngleDeg']:.0f}°")
        
        # Right: Bird's eye view
        ax2 = axes[1]
        ax2.plot(r['gt_traj'][:, 0], r['gt_traj'][:, 1], 'g-o', markersize=6, linewidth=2, label='GT')
        ax2.plot(r['pred_traj'][:, 0], r['pred_traj'][:, 1], 'r-o', markersize=6, linewidth=2, label='Pred')
        
        # Plot physics trajectory in bird's eye view
        if traj_physics is not None:
            ax2.plot(traj_physics[:, 0], traj_physics[:, 1], 'b-s', markersize=4, 
                     linewidth=1.5, label='Physics', alpha=0.7)
        
        ax2.scatter([0], [0], c='blue', s=150, marker='*', label='Ego', zorder=5)
        ax2.set_xlabel('Forward (m)')
        ax2.set_ylabel('Lateral (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.set_title("Bird's Eye View")
        
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

