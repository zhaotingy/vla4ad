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
    
    # Speed embedding (paper includes ego vehicle speed)
    use_speed_embedding: bool = True
    speed_embedding_dim: int = 64
    
    # Training
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 10
    
    # Loss weights (paper: equally weighted)
    caption_weight: float = 0.5
    trajectory_weight: float = 0.5
    
    # Data split (paper: 70/15/15)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
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
        
        # Filter and prepare samples
        self.samples = []
        
        for i in range(0, len(states_data), sample_interval):
            if i >= len(image_files):
                break
                
            state = states_data[i]
            trajectory = state.get('trajectory', [])
            
            # Paper: "excluding those lacking complete trajectory data for subsequent 3 seconds"
            if len(trajectory) < 60:
                continue
            
            # Uniformly sample 10 points from 60 (paper specification)
            traj_indices = np.linspace(0, 59, config.trajectory_points, dtype=int)
            sampled_trajectory = [trajectory[j] for j in traj_indices]
            
            # Get caption
            caption_idx = min(i // sample_interval, len(captions_data) - 1)
            caption = captions_data[caption_idx] if captions_data else {}
            
            # Get speed (required field - called 'vEgo' in dataset)
            if 'vEgo' not in state:
                raise KeyError(f"'vEgo' not found in state at index {i}. Available keys: {list(state.keys())}")
            
            self.samples.append({
                'image_path': image_files[i],
                'trajectory': sampled_trajectory,
                'caption': caption.get('rich_caption', caption.get('plain_caption', '')),
                'speed': state['vEgo'],
                'extrinsic_matrix': state['extrinsic_matrix'],
                'intrinsic_matrix': state['intrinsic_matrix'],
            })
        
        # Split data (70/15/15 as per paper)
        n = len(self.samples)
        train_end = int(n * config.train_ratio)
        val_end = train_end + int(n * config.val_ratio)
        
        if split == "train":
            self.samples = self.samples[:train_end]
        elif split == "val":
            self.samples = self.samples[train_end:val_end]
        else:  # test
            self.samples = self.samples[val_end:]
        
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
        
        # Speed (already extracted in __init__)
        speed = torch.tensor(sample.get('speed', 0.0), dtype=torch.float32)
        
        return {
            'image': image,
            'trajectory': trajectory,
            'caption': sample['caption'],
            'speed': speed,
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
            nn.Dropout(0.2),
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
        
        # Speed embedding MLP (paper: embeds ego vehicle speed)
        if config.use_speed_embedding:
            self.speed_embedding = nn.Sequential(
                nn.Linear(1, config.speed_embedding_dim),
                nn.ReLU(),
                nn.Linear(config.speed_embedding_dim, self.llm_dim),
            )
        else:
            self.speed_embedding = None
        
        # Trajectory MLP (paper specification)
        self.trajectory_mlp = TrajectoryMLP(
            input_dim=self.llm_dim,
            num_points=config.trajectory_points,
            coord_dim=config.trajectory_dim,
        )
        
        # Print model info
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_type = "PAPER" if config.use_paper_model else "LIGHTWEIGHT"
        print(f"✓ CoVLA-Agent initialized ({model_type})")
        print(f"  Vision: {config.vision_encoder}")
        print(f"  Language: {config.language_model}")
        print(f"  Speed embedding: {config.use_speed_embedding}")
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
        speeds: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both training and inference.
        
        Args:
            images: (batch, 3, H, W) input images
            captions: Captions for sequence (GT during training, predicted/GT during inference)
            trajectories: Ground truth trajectories (batch, 10, 3) - only for training
            speeds: Ego vehicle speeds (batch,) - REQUIRED
        
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
        
        # Speed embedding (paper: MLP that embeds ego vehicle speed) - 1 token
        speed_input = speeds.unsqueeze(-1).float()  # (batch, 1)
        speed_embeds = self.speed_embedding(speed_input)  # (batch, llm_dim)
        speed_embeds = speed_embeds.unsqueeze(1)  # (batch, 1, llm_dim)
        
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
        
        # During inference: generate captions if not provided
        # During training: captions should be GT captions (passed explicitly)
        if captions is None:
            with torch.no_grad():
                captions = self.generate_caption(images, speeds)
        
        caption_inputs = self.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        caption_embeds = self.language_model.get_input_embeddings()(caption_inputs.input_ids)
        caption_embeds = caption_embeds.to(prompt_embeds.dtype)
        
        # Sequence: [vision] + [speed] + [prompt] + [caption] + [traj_queries]
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
            
            # Combined loss (paper: equally weighted)
            result['loss'] = (
                self.config.caption_weight * caption_loss +
                self.config.trajectory_weight * trajectory_loss
            )
        
        return result
    
    @torch.no_grad()
    def generate_caption(
        self,
        images: torch.Tensor,
        speeds: torch.Tensor,
        max_length: int = 100,
    ) -> List[str]:
        """
        Generate driving scene captions conditioned on vision + speed.
        
        This properly uses the image by:
        1. Encoding image with CLIP
        2. Projecting to LLM space
        3. Using vision + speed tokens as prefix for generation
        """
        self.eval()
        device = images.device
        batch_size = images.shape[0]
        dtype = next(self.language_model.parameters()).dtype
        
        # 1. Encode image with CLIP and project to LLM space
        vision_features = self.encode_image(images)  # (B, num_patches, vision_dim)
        vision_embeds = self.vision_projection(vision_features).to(dtype)  # (B, num_patches, llm_dim)
        
        # 2. Add speed embedding (layer is float32, convert output to dtype)
        speed_embeds = self.speed_embedding(speeds.unsqueeze(-1).float().to(device))  # (B, llm_dim)
        speed_embeds = speed_embeds.to(dtype).unsqueeze(1)  # (B, 1, llm_dim)
        prefix_embeds = torch.cat([vision_embeds, speed_embeds], dim=1)
        
        # 3. Create prompt for generation
        prompt = "Describe the driving scene: "
        prompt_inputs = self.tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(device)
        prompt_embeds = self.language_model.get_input_embeddings()(prompt_inputs.input_ids).to(dtype)
        
        # 4. Combine: [Vision] + [Speed] + [Prompt]
        combined_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        attention_mask = torch.ones(batch_size, combined_embeds.shape[1], device=device)
        
        # 5. Generate using HuggingFace generate() with inputs_embeds
        outputs = self.language_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            do_sample=False,  # Greedy for consistency
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # 6. Decode generated tokens
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Clean up
        captions = [cap.strip() if cap.strip() else "The vehicle is driving on a road." for cap in captions]
        
        return captions
    
    @torch.no_grad()
    def predict(
        self, 
        image: torch.Tensor, 
        speed: float,
        caption: str = None,
        caption_mode: str = "pred",
    ) -> Dict:
        """
        Make prediction for single image.
        
        Paper Table 4 shows two inference modes with different ADE results:
        - "Pred. caption" (caption_mode="pred"): ADE 0.955 - generate caption, then predict trajectory
        - "GT caption" (caption_mode="gt"): ADE 0.814 - use GT caption for trajectory (oracle)
        
        Args:
            image: Input image tensor
            speed: Ego vehicle speed in m/s (REQUIRED)
            caption: Ground truth caption (required if caption_mode="gt")
            caption_mode: How to use captions for trajectory prediction
                - "pred": Generate caption first, use it for trajectory (default)
                - "gt": Use provided GT caption for trajectory (oracle mode, better ADE)
        
        Returns:
            - trajectory: (10, 3) predicted waypoints
            - caption: The caption used for prediction
        """
        self.eval()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        device = next(self.parameters()).device
        image = image.to(device)
        speeds = torch.tensor([speed], device=device)
        
        # Prepare caption based on mode
        if caption_mode == "gt":
            if caption is None:
                raise ValueError("caption_mode='gt' requires caption to be provided")
            caption_for_trajectory = caption
        else:
            # Pred caption mode: generate caption conditioned on image + speed
            generated_captions = self.generate_caption(image, speeds)
            caption_for_trajectory = generated_captions[0]
        
        # Get trajectory using the caption (either GT or predicted)
        output = self.forward(image, captions=[caption_for_trajectory], speeds=speeds)
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
        n_batches = 0
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            trajectories = batch['trajectory'].to(self.device)
            captions = batch['caption']  # GT captions for both conditioning and loss
            speeds = batch['speed'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Single forward pass: GT captions condition trajectory AND compute caption loss
            if self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.model(
                        images, 
                        captions=captions,          # GT captions for trajectory conditioning
                        trajectories=trajectories,
                        speeds=speeds,
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
                    speeds=speeds,
                )
                loss = output['loss']
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            total_traj_loss += output.get('trajectory_loss', torch.tensor(0)).item()
            total_caption_loss += output.get('caption_loss', torch.tensor(0)).item()
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'trajectory_loss': total_traj_loss / n_batches,
            'caption_loss': total_caption_loss / n_batches,
        }
    
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
            speeds = batch.get('speed')
            if speeds is not None:
                speeds = speeds.to(self.device)
            
            output = self.model(images, captions=captions, trajectories=trajectories, speeds=speeds)
            
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
        
        print("\n" + "=" * 70)
        print("Training CoVLA-Agent (Paper Implementation)")
        print("=" * 70)
        print(f"Train samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Val samples: {len(val_dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Loss weights: caption={self.config.caption_weight}, trajectory={self.config.trajectory_weight}")
        print("=" * 70 + "\n")
        
        header = f"{'Epoch':<8}{'Train Loss':<12}{'Val Loss':<12}{'ADE':<10}{'FDE':<10}"
        print(header)
        print("-" * len(header))
        
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
                
                print(f"{epoch+1:<8}{train_metrics['loss']:<12.4f}{val_metrics['loss']:<12.4f}"
                      f"{val_metrics['ade']:<10.3f}{val_metrics['fde']:<10.3f}")
            else:
                print(f"{epoch+1:<8}{train_metrics['loss']:<12.4f}")
        
        print("\n✓ Training complete!")
        
        # Print paper comparison
        if val_loader:
            final_ade = self.history['val_ade'][-1]
            final_fde = self.history['val_fde'][-1]
            print(f"\nFinal Results:")
            print(f"  ADE: {final_ade:.3f} (paper with predicted captions: 0.955)")
            print(f"  FDE: {final_fde:.3f} (paper with predicted captions: 2.239)")
        
        # Auto-save model
        save_path = "covla_model.pt"
        self.save_checkpoint(save_path)
        print(f"\n✓ Model saved to: {save_path}")
        
        return self.history
    
    def save_checkpoint(self, path: str = "covla_model.pt"):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to: {path}")
    
    def load_checkpoint(self, path: str = "covla_model.pt"):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded checkpoint from: {path}")
        return checkpoint.get('config')


def load_model(path: str = "covla_model.pt", device: str = "cuda") -> CoVLAAgentPaper:
    """
    Load a saved model checkpoint.
    
    Usage:
        model = load_model("covla_model.pt")
        result = model.predict(image, speed=speed, caption_mode="gt", caption=caption)
    """
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get('config', CoVLAConfig(device=device))
    
    model = CoVLAAgentPaper(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from: {path}")
    if 'history' in checkpoint and checkpoint['history'].get('val_ade'):
        print(f"  Last ADE: {checkpoint['history']['val_ade'][-1]:.3f}m")
    
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
               marker='o', color=color, linestyle='solid', 
               linewidth=2, markersize=4, alpha=0.8, label=label)
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
    axes[0].set_title(f"idx={sample_idx} | {video_id}/{frame_name} | {speed:.1f} m/s")
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
    print(f"🚗 Speed: {speed:.1f} m/s")
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
    speed = sample['speed'].item() if hasattr(sample['speed'], 'item') else float(sample['speed'])
    caption = sample.get('caption', 'No caption')
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Image with trajectory
    plot_trajectory_on_image(frame, trajectory, extrinsic, intrinsic, 
                             color='lime', label='GT Trajectory', ax=axes[0])
    axes[0].set_title(f"Dataset sample {sample_idx} | {speed:.1f} m/s")
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
    print(f"🚗 Speed: {speed:.1f} m/s")
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
    speed: float,
    gt_caption: str = None,
    image_path: str = None,
    caption_mode: str = "pred",
):
    """
    Visualize model inference with trajectory overlay on image.
    
    Following the style of:
    https://huggingface.co/datasets/turing-motors/CoVLA-Dataset/blob/main/tutorial.ipynb
    
    Paper Table 4 - Two inference modes:
    - caption_mode="pred": Generate caption → use for trajectory (ADE ~0.955)
    - caption_mode="gt":   Use GT caption → trajectory (ADE ~0.814, oracle)
    
    Args:
        model: CoVLAAgentPaper model
        image: Input image tensor (3, H, W) or (1, 3, H, W)
        gt_trajectory: Ground truth trajectory (N, 3)
        extrinsic_matrix: Camera extrinsic matrix
        intrinsic_matrix: Camera intrinsic matrix
        gt_caption: Ground truth caption (required if caption_mode="gt")
        image_path: Path to original image file (optional, for high-res display)
        speed: Ego vehicle speed in m/s (important if model trained with speed embedding)
        caption_mode: "none" (default) or "gt" - see Paper Table 4
    """
    import matplotlib.pyplot as plt
    
    # Get model prediction based on caption mode
    result = model.predict(
        image, 
        speed=speed, 
        caption=gt_caption,
        caption_mode=caption_mode,
    )
    pred_traj = result['trajectory']
    pred_caption = result['caption']
    
    # Load high-res image if path provided, else convert tensor
    if image_path and os.path.exists(image_path):
        frame = np.array(Image.open(image_path))
        if frame.shape[2] == 4:  # RGBA
            frame = frame[:, :, :3]
    else:
        # Convert tensor to numpy
        if image.dim() == 4:
            image = image[0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame = ((image.cpu() * std + mean).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Compute metrics
    pred_tensor = torch.tensor(pred_traj).unsqueeze(0)
    gt_tensor = torch.tensor(gt_trajectory).unsqueeze(0)
    ade = compute_ade(pred_tensor, gt_tensor)
    fde = compute_fde(pred_tensor, gt_tensor)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Top: Image with trajectory overlay
    ax1 = fig.add_subplot(2, 2, 1)
    plot_trajectory_on_image(
        frame, gt_trajectory, extrinsic_matrix, intrinsic_matrix,
        color='green', label='Ground Truth', ax=ax1
    )
    plot_trajectory_on_image(
        frame, pred_traj, extrinsic_matrix, intrinsic_matrix,
        color='red', label='Predicted', ax=ax1
    )
    ax1.legend(loc='upper right')
    ax1.set_title("Trajectory on Image", fontsize=12)
    
    # Top right: Bird's eye view
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'g-o', 
            markersize=5, linewidth=2, label='Ground Truth')
    ax2.plot(pred_traj[:, 0], pred_traj[:, 1], 'r-o',
            markersize=5, linewidth=2, label='Predicted')
    ax2.scatter([0], [0], c='blue', s=100, marker='*', label='Ego', zorder=5)
    ax2.set_xlabel('Forward (m)')
    ax2.set_ylabel('Lateral (m)')
    ax2.set_title(f"Bird's Eye View\nADE: {ade:.2f}m | FDE: {fde:.2f}m", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Bottom: Captions
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.axis('off')
    
    caption_text = f"""🤖 Generated Caption:
{pred_caption[:400]}{'...' if len(pred_caption) > 400 else ''}

"""
    if gt_caption:
        caption_text += f"""📝 Ground Truth Caption:
{gt_caption[:400]}{'...' if len(gt_caption) > 400 else ''}

"""
    caption_text += f"""📊 Metrics:
  • ADE (Average Displacement Error): {ade:.3f} m
  • FDE (Final Displacement Error): {fde:.3f} m"""
    
    ax3.text(0.02, 0.95, caption_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='sans-serif',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=10),
             wrap=True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'pred_trajectory': pred_traj,
        'pred_caption': pred_caption,
        'ade': ade,
        'fde': fde,
    }


def visualize_prediction_paper(
    model: CoVLAAgentPaper,
    sample: Dict,
    state: Dict = None,
    image_path: str = None,
):
    """
    Visualize prediction vs ground truth (simplified interface).
    
    Args:
        model: CoVLAAgentPaper model
        sample: Dataset sample with 'image', 'trajectory', 'caption'
        state: Original state dict with camera matrices (optional)
        image_path: Path to original image (optional)
    """
    import matplotlib.pyplot as plt
    
    # Get trajectories
    gt_traj = sample['trajectory'].numpy() if torch.is_tensor(sample['trajectory']) else sample['trajectory']
    
    # Check if we have camera matrices
    if state and 'extrinsic_matrix' in state and 'intrinsic_matrix' in state:
        extrinsic = np.array(state['extrinsic_matrix'])
        intrinsic = np.array(state['intrinsic_matrix'])
        
        return visualize_inference(
            model=model,
            image=sample['image'],
            gt_trajectory=gt_traj,
            extrinsic_matrix=extrinsic,
            intrinsic_matrix=intrinsic,
            gt_caption=sample.get('caption') or sample.get('plain_caption'),
            image_path=image_path,
        )
    else:
        # Fallback: just show bird's eye view and caption
        result = model.predict(sample['image'])
        pred_traj = result['trajectory']
        
        # Compute metrics
        pred_tensor = torch.tensor(pred_traj).unsqueeze(0)
        gt_tensor = torch.tensor(gt_traj).unsqueeze(0)
        ade = compute_ade(pred_tensor, gt_tensor)
        fde = compute_fde(pred_tensor, gt_tensor)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Bird's eye view
        ax1 = axes[0]
        ax1.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-o', markersize=6, 
                linewidth=2, label='Ground Truth')
        ax1.plot(pred_traj[:, 0], pred_traj[:, 1], 'r-o', markersize=6,
                linewidth=2, label='Predicted')
        ax1.scatter([0], [0], c='blue', s=150, marker='*', label='Ego', zorder=5)
        ax1.set_xlabel('Forward (m)')
        ax1.set_ylabel('Lateral (m)')
        ax1.set_title(f"Trajectory (Bird's Eye View)\nADE: {ade:.2f}m | FDE: {fde:.2f}m")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Right: Caption
        ax2 = axes[1]
        ax2.axis('off')
        
        caption_text = f"""🤖 Generated Caption:
{result['caption'][:300]}...

📊 Metrics:
  • ADE: {ade:.3f} m
  • FDE: {fde:.3f} m"""
        
        ax2.text(0.05, 0.95, caption_text, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        plt.show()
        
        return {'pred_trajectory': pred_traj, 'pred_caption': result['caption'], 'ade': ade, 'fde': fde}


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

