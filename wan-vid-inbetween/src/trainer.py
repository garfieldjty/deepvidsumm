from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from pathlib import Path
import json
import os
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from tqdm.auto import tqdm

from .utils_latents import retrieve_latents
from .models import load_wan_components, add_lora_to_transformer
from .dataset import InbetweenVideoDataset


class CumulativeSoftmaxFusionMLP(nn.Module):
    """
    MLP that produces per-frame fusion weights using Cumulative Softmax.
    
    The Cumulative Softmax ensures monotonic blending weights that smoothly
    transition from forward-generated frames (early) to backward-generated 
    frames (later in the sequence).
    
    Given T_mid latent frames, produces weights w_fwd[t] and w_bwd[t] where:
    - w_fwd + w_bwd = 1 (per frame)
    - w_fwd is monotonically decreasing (or non-increasing)
    - w_bwd is monotonically increasing (or non-decreasing)
    
    Uses CNN-based spatial downscaling instead of aggressive global pooling
    to preserve more spatial information for better weight prediction.
    
    The MLP is trained by comparing predicted weights against ground-truth
    latent similarity patterns (how similar each mid frame is to start vs end).
    """
    
    def __init__(
        self, 
        latent_dim: int,  # Channel dimension of latents (e.g., 16 for Wan VAE)
        hidden_dim: int = 256,
        num_layers: int = 3,
        max_frames: int = 64,  # Maximum number of mid frames to support
        cnn_feature_dim: int = 64,  # Output feature dim from CNN encoder
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_frames = max_frames
        self.cnn_feature_dim = cnn_feature_dim
        
        # CNN encoder to downsample spatial dimensions while preserving info
        # Input: [B, C, H, W] -> Output: [B, cnn_feature_dim]
        # Uses strided convolutions for progressive downsampling
        self.spatial_encoder = nn.Sequential(
            # First conv: C -> 32, reduce spatial by 2x
            nn.Conv2d(latent_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            # Second conv: 32 -> 64, reduce spatial by 2x
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Third conv: 64 -> 64, reduce spatial by 2x
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Adaptive pool to fixed size (2x2) then flatten
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),  # [B, 64 * 2 * 2] = [B, 256]
            # Project to cnn_feature_dim
            nn.Linear(64 * 4, cnn_feature_dim),
            nn.SiLU(),
        )
        
        # Input to MLP: [start_features, end_features, position_encoding]
        # start/end features: cnn_feature_dim each
        # position encoding: cnn_feature_dim (to match feature dimensions)
        input_dim = cnn_feature_dim * 3
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        # Output: logits for cumulative softmax (one per frame)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Pre-compute position encoding frequencies
        # Use cnn_feature_dim for position encoding to match CNN features
        self.register_buffer(
            'pos_freqs',
            1.0 / (10000.0 ** (torch.arange(0, cnn_feature_dim, 2).float() / cnn_feature_dim))
        )
        self.pos_scale = cnn_feature_dim ** 0.5
        
    def _get_position_encodings_batch(self, T_mid: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generate sinusoidal position encodings for all T_mid frames at once.
        
        Returns:
            pos_enc: [T_mid, cnn_feature_dim] position encodings
        """
        # Normalized positions in [0, 1]
        positions = torch.linspace(0, 1, T_mid, device=device, dtype=dtype)  # [T_mid]
        
        # Compute angles: [T_mid, cnn_feature_dim // 2]
        freqs = self.pos_freqs.to(device=device, dtype=dtype)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0) * self.pos_scale  # [T_mid, D/2]
        
        # Build position encoding
        pe = torch.zeros(T_mid, self.cnn_feature_dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, :self.cnn_feature_dim // 2])
        
        return pe
    
    def forward(
        self,
        start_lat: torch.Tensor,  # [B, C, T_start, H, W]
        end_lat: torch.Tensor,    # [B, C, T_end, H, W]
        T_mid: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fusion weights for each mid frame.
        
        Returns:
            w_fwd: [B, T_mid] weights for forward-generated latents
            w_bwd: [B, T_mid] weights for backward-generated latents
        """
        B, C, _, H, W = start_lat.shape
        device = start_lat.device
        dtype = start_lat.dtype
        
        # Extract boundary frames: [B, C, H, W]
        start_last_frame = start_lat[:, :, -1]  # last frame of start
        end_first_frame = end_lat[:, :, 0]      # first frame of end
        
        # CNN encode boundary frames: [B, C, H, W] -> [B, cnn_feature_dim]
        start_features = self.spatial_encoder(start_last_frame)  # [B, cnn_feature_dim]
        end_features = self.spatial_encoder(end_first_frame)      # [B, cnn_feature_dim]
        
        # Get all position encodings at once: [T_mid, cnn_feature_dim]
        pos_enc = self._get_position_encodings_batch(T_mid, device, dtype)
        
        # Expand for batch: [B, T_mid, cnn_feature_dim]
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)
        
        # Expand boundary features: [B, T_mid, cnn_feature_dim]
        start_features_exp = start_features.unsqueeze(1).expand(-1, T_mid, -1)
        end_features_exp = end_features.unsqueeze(1).expand(-1, T_mid, -1)
        
        # Concatenate features: [B, T_mid, 3 * cnn_feature_dim]
        feat = torch.cat([start_features_exp, end_features_exp, pos_enc], dim=-1)
        
        # Reshape for batch MLP: [B * T_mid, 3 * cnn_feature_dim]
        feat_flat = feat.view(B * T_mid, -1)
        
        # MLP forward: [B * T_mid, 1]
        logits_flat = self.mlp(feat_flat)
        
        # Reshape back: [B, T_mid]
        logits = logits_flat.view(B, T_mid)
        
        # Cumulative Softmax for monotonic weights
        # w_bwd[t] = cumsum(softmax(logits))[t]
        probs = torch.softmax(logits, dim=-1)  # [B, T_mid]
        w_bwd = torch.cumsum(probs, dim=-1)    # [B, T_mid], monotonically increasing
        w_fwd = 1.0 - w_bwd                     # [B, T_mid], monotonically decreasing
        
        return w_fwd, w_bwd
    
    @staticmethod
    def compute_gt_weights_from_similarity(
        mid_lat: torch.Tensor,    # [B, C, T_mid, H, W] - ground truth mid latents
        start_lat: torch.Tensor,  # [B, C, T_start, H, W]
        end_lat: torch.Tensor,    # [B, C, T_end, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ground-truth fusion weights based on cosine similarity.
        
        For each mid frame, compute similarity to:
        - Last frame of start segment (sim_start)
        - First frame of end segment (sim_end)
        
        GT weights: w_fwd_gt = sim_start / (sim_start + sim_end + eps)
        
        Returns:
            w_fwd_gt: [B, T_mid] ground-truth forward weights
            w_bwd_gt: [B, T_mid] ground-truth backward weights
        """
        B, C, T_mid, H, W = mid_lat.shape
        
        # Flatten spatial dimensions for cosine similarity
        # mid: [B, T_mid, C*H*W]
        mid_flat = mid_lat.permute(0, 2, 1, 3, 4).reshape(B, T_mid, -1)
        
        # Reference frames: [B, C*H*W]
        start_ref = start_lat[:, :, -1].reshape(B, -1)  # last frame of start
        end_ref = end_lat[:, :, 0].reshape(B, -1)       # first frame of end
        
        # Normalize for cosine similarity
        mid_norm = torch.nn.functional.normalize(mid_flat, dim=-1)
        start_norm = torch.nn.functional.normalize(start_ref, dim=-1).unsqueeze(1)  # [B, 1, CHW]
        end_norm = torch.nn.functional.normalize(end_ref, dim=-1).unsqueeze(1)      # [B, 1, CHW]
        
        # Cosine similarity: [B, T_mid]
        sim_start = (mid_norm * start_norm).sum(dim=-1)  # similarity to start
        sim_end = (mid_norm * end_norm).sum(dim=-1)      # similarity to end
        
        # Convert to weights (shifted to [0, 1] range since cosine can be negative)
        # Use softmax-style normalization
        sim_start_pos = torch.clamp(sim_start, min=0.0) + 0.1  # add small constant
        sim_end_pos = torch.clamp(sim_end, min=0.0) + 0.1
        
        w_fwd_gt = sim_start_pos / (sim_start_pos + sim_end_pos)
        w_bwd_gt = 1.0 - w_fwd_gt
        
        return w_fwd_gt, w_bwd_gt

# Enable cuDNN benchmarking for faster convolutions
torch.backends.cudnn.benchmark = True

# Enable TF32 for Ampere GPUs (3x speedup on matmul/conv)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class TrainConfig:
    base_model_path: str
    data_root: str
    video_glob: str
    clip_num_frames: int
    height: int
    width: int
    start_frames: int
    mid_frames: int
    end_frames: int
    cut_annotations_path: str
    use_cut_focused_sampling: bool
    cut_focus_prob: float
    train_batch_size: int
    gradient_accumulation_steps: int
    num_train_steps: int
    learning_rate: float
    seed: int
    output_dir: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    transformer_precision: str
    vae_precision: str
    # Performance
    attn_implementation: str = "sdpa"  # "flash_attention_2", "sdpa", or "eager"
    # Logging
    log_dir: str = "./logs"
    log_every_n_steps: int = 10
    # Checkpointing
    save_every_n_steps: int = 1000
    resume_from_checkpoint: Optional[str] = None


class InbetweenTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        
        # Enable cuDNN optimizations for faster training
        torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 on Ampere+ GPUs (3x speedup)
        torch.backends.cudnn.allow_tf32 = True
        
        self.acc = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision="bf16" if cfg.transformer_precision == "bf16" else "fp16"
        )

        # Load components (custom Wan transformer with conditioning mask)
        self.vae, self.transformer, self.scheduler = load_wan_components(
            cfg.base_model_path,
            transformer_precision=cfg.transformer_precision,
            vae_precision=cfg.vae_precision,
            attn_implementation=cfg.attn_implementation,
        )

        # Add LoRA
        self.transformer = add_lora_to_transformer(
            self.transformer,
            cfg.lora_r,
            cfg.lora_alpha,
            cfg.lora_dropout,
            cfg.num_train_steps,
        )

        # Freeze all except LoRA
        for name, p in self.transformer.named_parameters():
            p.requires_grad = ("lora_" in name)

        params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optim = torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=1e-5)

        # Dataset
        dataset = InbetweenVideoDataset(
            data_root=cfg.data_root,
            video_glob=cfg.video_glob,
            clip_num_frames=cfg.clip_num_frames,
            height=cfg.height,
            width=cfg.width,
            start_frames=cfg.start_frames,
            mid_frames=cfg.mid_frames,
            end_frames=cfg.end_frames,
            cut_annotations_path=cfg.cut_annotations_path,
            use_cut_focused_sampling=cfg.use_cut_focused_sampling,
            cut_focus_prob=cfg.cut_focus_prob,
        )

        self.dl = DataLoader(
            dataset, 
            batch_size=cfg.train_batch_size, 
            shuffle=True,
            num_workers=8,  # Increase parallel data loading
            pin_memory=True,  # Faster CPU->GPU transfer
            prefetch_factor=4,  # Prefetch more batches
            persistent_workers=True  # Keep workers alive
        )

        self.transformer, self.optim, self.dl = self.acc.prepare(
            self.transformer, self.optim, self.dl
        )
        # Move VAE to device manually
        self.vae = self.vae.to(self.acc.device)
        
        # Cache unwrapped transformer reference for accessing config/dtype
        self._unwrapped_transformer = self.acc.unwrap_model(self.transformer)

        # Pre-compute latent normalization tensors (avoid recreating every step)
        self.latents_mean = None
        self.latents_std = None

        # Log distributed training info
        if self.acc.is_local_main_process:
            num_gpus = self.acc.num_processes
            effective_batch = cfg.train_batch_size * num_gpus * cfg.gradient_accumulation_steps
            print(f"Training on {num_gpus} GPU(s)")
            print(f"  Per-GPU batch size: {cfg.train_batch_size}")
            print(f"  Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
            print(f"  Effective batch size: {effective_batch}")
            print(f"  Attention: {cfg.attn_implementation}")
            print(f"  cuDNN benchmark: enabled (TF32: {torch.backends.cuda.matmul.allow_tf32})")
            
            # Warn about small batch sizes with multi-GPU
            if num_gpus > 1 and cfg.train_batch_size == 1:
                print("\n⚠️  WARNING: Per-GPU batch size of 1 is too small for multi-GPU training!")
                print("   Multi-GPU overhead negates speedup. Increase train_batch_size to 2-4 per GPU.")
                print(f"   Recommended: train_batch_size >= 2, reduce gradient_accumulation_steps to {cfg.gradient_accumulation_steps // 2}")
                print(f"   This keeps effective batch size at {effective_batch} but improves GPU utilization.\n")

        # TensorBoard logging
        self.writer = None
        if self.acc.is_local_main_process:
            log_dir = Path(cfg.log_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard logs → {log_dir}")

        # Checkpoint directory
        self.checkpoint_dir = Path(cfg.output_dir) / "checkpoints"
        if self.acc.is_local_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track global step (may be restored from checkpoint)
        self.global_step = 0


    def _sample_timesteps(self, batch_size: int, device: torch.device, timestep_sampling_power: float) -> torch.Tensor:
        """
        Stratified + power-biased sampling over [0, T).

        - Stratification reduces variance across the batch.
        - `timestep_sampling_power` lets you bias towards early or late times:
            * 1.0  -> uniform
            * 0.5  -> focus more on small t (cleaner/noisier mid region)
            * 2.0  -> focus more on large t (very noisy region)
        """
        T = self.scheduler.config.num_train_timesteps

        # stratified samples in [0, 1)
        u = torch.rand(batch_size, device=device)
        strata = (torch.arange(batch_size, device=device, dtype=u.dtype) + u) / batch_size  # (0,1)

        # optional bias in log-space of t
        p = timestep_sampling_power
        if p != 1.0:
            strata = strata ** p  # p<1 -> concentrates near 0, p>1 -> concentrates near 1

        # map to [0, T)
        t = strata * T
        return t


    def save_checkpoint(self, step: int):
        """Save a checkpoint for pause/resume."""
        if not self.acc.is_local_main_process:
            return

        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        unwrapped = self.acc.unwrap_model(self.transformer)
        unwrapped.save_pretrained(checkpoint_path / "lora")

        # Save optimizer state
        torch.save(self.optim.state_dict(), checkpoint_path / "optimizer.pt")

        # Save training state
        state = {
            "global_step": step,
            "config": {
                "learning_rate": self.cfg.learning_rate,
                "num_train_steps": self.cfg.num_train_steps,
                "lora_r": self.cfg.lora_r,
                "lora_alpha": self.cfg.lora_alpha,
            }
        }
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"Checkpoint saved → {checkpoint_path}")

        # Keep only last 3 checkpoints to save disk space
        self._cleanup_old_checkpoints(keep=3)

    def _cleanup_old_checkpoints(self, keep: int = 3):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1])
        )
        for ckpt in checkpoints[:-keep]:
            import shutil
            shutil.rmtree(ckpt)
            print(f"Removed old checkpoint: {ckpt}")

    def load_checkpoint(self, checkpoint_path_str: str):
        """Load a checkpoint to resume training."""
        checkpoint_path = Path(checkpoint_path_str)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load training state
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
            self.global_step = state["global_step"]
            print(f"Resuming from step {self.global_step}")

        # Load LoRA weights
        lora_path = checkpoint_path / "lora"
        if lora_path.exists():
            from peft import PeftModel
            # The transformer already has LoRA, we need to load the weights
            unwrapped = self.acc.unwrap_model(self.transformer)
            unwrapped.load_adapter(str(lora_path), adapter_name="default")
            print(f"Loaded LoRA weights from {lora_path}")

        # Load optimizer state
        optim_path = checkpoint_path / "optimizer.pt"
        if optim_path.exists():
            optim_state = torch.load(optim_path, map_location=self.acc.device)
            self.optim.load_state_dict(optim_state)
            print(f"Loaded optimizer state from {optim_path}")

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint in the checkpoint directory."""
        if not self.checkpoint_dir.exists():
            return None
        
        checkpoints = list(self.checkpoint_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None
        
        # Sort by step number and return the latest
        latest = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
        return latest

    # ------------------------------------------------------------
    # TRAINING STEP (mask-based [start, mid, end] conditioning)
    # ------------------------------------------------------------
    def train(self):
        device = self.acc.device
        self.transformer.train()
        self.vae.eval()

        # Resume from checkpoint if specified
        if self.cfg.resume_from_checkpoint:
            if self.cfg.resume_from_checkpoint == "latest":
                latest = self.get_latest_checkpoint()
                if latest:
                    self.load_checkpoint(str(latest))
                else:
                    print("No checkpoint found, starting from scratch")
            else:
                self.load_checkpoint(self.cfg.resume_from_checkpoint)

        # Start from global_step (may have been restored from checkpoint)
        step = self.global_step
        remaining_steps = self.cfg.num_train_steps - step
        
        if remaining_steps <= 0:
            print(f"Training already completed ({step}/{self.cfg.num_train_steps} steps)")
            return

        pbar = tqdm(
            total=self.cfg.num_train_steps,
            initial=step,
            disable=not self.acc.is_local_main_process,
            desc="Training"
        )

        # For tracking running loss
        running_loss = 0.0
        loss_count = 0

        while step < self.cfg.num_train_steps:
            for batch in self.dl:
                with self.acc.accumulate(self.transformer):

                    # Video: [B, T, C, H, W] -> [B, C, T, H, W]
                    video = batch["video"].to(device, non_blocking=True)
                    video = video.permute(0, 2, 1, 3, 4)
                    B = video.shape[0]

                    # Split video into start/mid/end segments in pixel space
                    s_frames = self.cfg.start_frames
                    m_frames = self.cfg.mid_frames
                    e_frames = self.cfg.end_frames
                    
                    video_start = video[:, :, :s_frames]           # [B, C, s_frames, H, W]
                    video_mid = video[:, :, s_frames:s_frames+m_frames]  # [B, C, m_frames, H, W]
                    video_end = video[:, :, s_frames+m_frames:]    # [B, C, e_frames, H, W]

                    # Encode each segment SEPARATELY via VAE (matches inference)
                    with torch.no_grad():
                        # Encode start
                        enc_start = self.vae.encode(video_start)
                        start_lat = retrieve_latents(enc_start)
                        
                        # Encode mid
                        enc_mid = self.vae.encode(video_mid)
                        mid_lat = retrieve_latents(enc_mid)
                        
                        # Encode end
                        enc_end = self.vae.encode(video_end)
                        end_lat = retrieve_latents(enc_end)
                        
                        # Convert to transformer dtype
                        start_lat = start_lat.to(self._unwrapped_transformer.dtype)
                        mid_lat = mid_lat.to(self._unwrapped_transformer.dtype)
                        end_lat = end_lat.to(self._unwrapped_transformer.dtype)

                        # Normalize latents using Wan's config (cached for efficiency)
                        if self.latents_mean is None:
                            self.latents_mean = torch.tensor(
                                self.vae.config.latents_mean,
                                device=start_lat.device,
                                dtype=start_lat.dtype,
                            ).view(1, self.vae.config.z_dim, 1, 1, 1)
                            self.latents_std = 1.0 / torch.tensor(
                                self.vae.config.latents_std,
                                device=start_lat.device,
                                dtype=start_lat.dtype,
                            ).view(1, self.vae.config.z_dim, 1, 1, 1)
                        
                        start_lat = (start_lat - self.latents_mean) * self.latents_std
                        mid_lat = (mid_lat - self.latents_mean) * self.latents_std
                        end_lat = (end_lat - self.latents_mean) * self.latents_std

                    # Get latent temporal dimensions
                    _, C, T_start_lat, H, W = start_lat.shape
                    _, _, T_mid_lat, _, _ = mid_lat.shape
                    _, _, T_end_lat, _, _ = end_lat.shape
                    T_lat = T_start_lat + T_mid_lat + T_end_lat
                    
                    assert T_start_lat > 0 and T_mid_lat > 0 and T_end_lat > 0, "Invalid latent splits"

                    # Noise & timestep (only for mid)
                    noise = torch.randn_like(mid_lat)

                    t = self._sample_timesteps(B, device, 0.5)
                    
                    # Flow matching interpolation: x_t = (1 - t/T) * x_0 + (t/T) * noise
                    # where T = num_train_timesteps (typically 1000)
                    t_normalized = t.view(B, 1, 1, 1, 1) / self.scheduler.config.num_train_timesteps
                    noisy_mid = (1 - t_normalized) * mid_lat + t_normalized * noise

                    # Build transformer input: [start, noisy_mid, end]
                    latents_in = torch.cat([start_lat, noisy_mid, end_lat], dim=2)

                    # Conditioning mask at LATENT temporal level: True for start+end, False for mid
                    conditioning_mask = torch.zeros(
                        B, T_lat, device=device, dtype=torch.bool
                    )
                    conditioning_mask[:, :T_start_lat] = True
                    conditioning_mask[:, T_start_lat + T_mid_lat:] = True

                    # Dummy text embeddings (unconditional)
                    # Access config from cached unwrapped model
                    enc_state = torch.zeros(
                        B, 1, self._unwrapped_transformer.config.text_dim,
                        device=device,
                        dtype=self._unwrapped_transformer.dtype,
                    )

                    # Forward
                    pred = self.transformer(
                        hidden_states=latents_in,
                        timestep=t,
                        encoder_hidden_states=enc_state,
                        conditioning_mask=conditioning_mask,
                        return_dict=True,
                    ).sample  # [B, C, T_lat, H, W]

                    # Only supervise mid region (using latent temporal indices)
                    pred_mid = pred[:, :, T_start_lat:T_start_lat + T_mid_lat]
                    # Flow matching target: velocity field (noise - x_0)
                    target = noise - mid_lat
                    loss = torch.nn.functional.mse_loss(pred_mid.float(), target.float())

                    # lambda_bound = 0.1  # tune

                    # # in clean latent space (x0), boundaries are:
                    # mid_first_clean = mid_lat[:, :, 0]          # latent at start of mid
                    # mid_last_clean  = mid_lat[:, :, -1]         # latent at end of mid
                    # start_last      = start_lat[:, :, -1]
                    # end_first       = end_lat[:, :, 0]

                    # boundary_loss = torch.nn.functional.mse_loss(mid_first_clean, start_last) + \
                    #                 torch.nn.functional.mse_loss(mid_last_clean,  end_first)

                    # loss = loss + lambda_bound * boundary_loss

                    self.acc.backward(loss)
                    self.optim.step()
                    self.optim.zero_grad()

                    step += 1
                    pbar.update(1)

                    # Track loss for logging (gather from all processes)
                    # Use .item() before gather to avoid memory issues
                    loss_value = loss.detach().item()
                    running_loss += loss_value
                    loss_count += 1

                    # TensorBoard logging (only on main process)
                    # Note: Each GPU logs its own local loss, not gathered across GPUs
                    # This is intentional to avoid NCCL sync issues during logging
                    if self.writer and step % self.cfg.log_every_n_steps == 0:
                        avg_loss = running_loss / loss_count
                        if self.acc.is_local_main_process:
                            self.writer.add_scalar("train/loss", avg_loss, step)
                            self.writer.add_scalar("train/learning_rate", self.cfg.learning_rate, step)
                            self.writer.add_scalar("train/timestep_mean", t.mean().item(), step)
                            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                        running_loss = 0.0
                        loss_count = 0

                    # Save checkpoint periodically
                    if step % self.cfg.save_every_n_steps == 0:
                        # Wait for all processes to reach this point
                        self.acc.wait_for_everyone()
                        self.save_checkpoint(step)

                    if step >= self.cfg.num_train_steps:
                        break
        
        # Update global step for potential resume
        self.global_step = step

        # Close progress bar
        pbar.close()

        # Final save - wait for all processes
        self.acc.wait_for_everyone()
        
        if self.acc.is_local_main_process:
            # Save final checkpoint
            self.save_checkpoint(step)
            
            # Also save to output_dir for easy access
            output_path = Path(self.cfg.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self.acc.unwrap_model(self.transformer).save_pretrained(output_path)
            print(f"Saved final LoRA → {output_path}")

            # Close TensorBoard writer
            if self.writer:
                self.writer.close()
        
        # Final barrier to ensure all processes finish together
        self.acc.wait_for_everyone()


# =============================================================================
# BIDIRECTIONAL INBETWEENING TRAINER
# =============================================================================
# This trainer uses two separate DiT passes (forward continuation + backward
# continuation) and learns to fuse them via a Cumulative Softmax MLP.
# =============================================================================

@dataclass
class BidirectionalTrainConfig:
    """Configuration for bidirectional inbetweening training."""
    base_model_path: str
    data_root: str
    video_glob: str
    clip_num_frames: int
    height: int
    width: int
    start_frames: int
    mid_frames: int
    end_frames: int
    cut_annotations_path: str
    use_cut_focused_sampling: bool
    cut_focus_prob: float
    train_batch_size: int
    gradient_accumulation_steps: int
    num_train_steps: int
    learning_rate: float
    seed: int
    output_dir: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    transformer_precision: str
    vae_precision: str
    # Performance
    attn_implementation: str = "sdpa"
    # Logging
    log_dir: str = "./logs"
    log_every_n_steps: int = 10
    # Checkpointing
    save_every_n_steps: int = 1000
    resume_from_checkpoint: Optional[str] = None
    # Fusion MLP settings
    fusion_hidden_dim: int = 256
    fusion_num_layers: int = 3
    cnn_feature_dim: int = 64  # Output feature dim from CNN spatial encoder
    pretrained_fusion_mlp_path: Optional[str] = None  # Path to pre-trained fusion MLP weights
    # Alternating training settings
    alternating_steps: int = 200  # Train each LoRA for this many steps before switching


class BidirectionalInbetweenTrainer:
    """
    Trainer for bidirectional inbetweening with fusion.
    
    This approach:
    1. Trains a "forward" LoRA to continue from start frames
    2. Trains a "backward" LoRA to generate missing beginning for end frames
    3. Uses a pre-trained Cumulative Softmax MLP to fuse the two predictions at inference
    
    Training alternates between the two LoRAs every N steps to avoid gradient conflicts.
    """
    
    def __init__(self, cfg: BidirectionalTrainConfig):
        self.cfg = cfg
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.acc = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision="bf16" if cfg.transformer_precision == "bf16" else "fp16"
        )

        # Load base components
        self.vae, transformer_base, self.scheduler = load_wan_components(
            cfg.base_model_path,
            transformer_precision=cfg.transformer_precision,
            vae_precision=cfg.vae_precision,
            attn_implementation=cfg.attn_implementation,
        )

        # Create FORWARD LoRA
        self.transformer_fwd = add_lora_to_transformer(
            transformer_base,
            cfg.lora_r,
            cfg.lora_alpha,
            cfg.lora_dropout,
            cfg.num_train_steps,
        )
        
        # Create a fresh copy of the base transformer for backward LoRA
        # Need to reload to get a separate instance
        _, transformer_base_bwd, _ = load_wan_components(
            cfg.base_model_path,
            transformer_precision=cfg.transformer_precision,
            vae_precision=cfg.vae_precision,
            attn_implementation=cfg.attn_implementation,
        )
        
        # Create BACKWARD LoRA
        self.transformer_bwd = add_lora_to_transformer(
            transformer_base_bwd,
            cfg.lora_r,
            cfg.lora_alpha,
            cfg.lora_dropout,
            cfg.num_train_steps,
        )

        # Freeze all except LoRA for both transformers
        for name, p in self.transformer_fwd.named_parameters():
            p.requires_grad = ("lora_" in name)
        for name, p in self.transformer_bwd.named_parameters():
            p.requires_grad = ("lora_" in name)

        # Get latent channel dimension from VAE config
        latent_dim = self.vae.config.z_dim
        
        # Create fusion MLP with CNN spatial encoder
        self.fusion_mlp = CumulativeSoftmaxFusionMLP(
            latent_dim=latent_dim,
            hidden_dim=cfg.fusion_hidden_dim,
            num_layers=cfg.fusion_num_layers,
            cnn_feature_dim=cfg.cnn_feature_dim,
        )
        
        # Load pre-trained fusion MLP if provided
        self.fusion_mlp_frozen = False
        if cfg.pretrained_fusion_mlp_path:
            mlp_weights = torch.load(cfg.pretrained_fusion_mlp_path, map_location="cpu")
            self.fusion_mlp.load_state_dict(mlp_weights)
            # Freeze MLP parameters
            for p in self.fusion_mlp.parameters():
                p.requires_grad = False
            self.fusion_mlp_frozen = True
            if self.acc.is_local_main_process:
                print(f"Loaded pre-trained fusion MLP from {cfg.pretrained_fusion_mlp_path} (frozen)")

        # Collect trainable parameters for each LoRA
        params_fwd = [p for p in self.transformer_fwd.parameters() if p.requires_grad]
        params_bwd = [p for p in self.transformer_bwd.parameters() if p.requires_grad]
        
        # Create separate optimizers for each LoRA
        self.optim_fwd = torch.optim.Adam(params_fwd, lr=cfg.learning_rate, weight_decay=1e-5)
        self.optim_bwd = torch.optim.Adam(params_bwd, lr=cfg.learning_rate, weight_decay=1e-5)

        # Dataset
        dataset = InbetweenVideoDataset(
            data_root=cfg.data_root,
            video_glob=cfg.video_glob,
            clip_num_frames=cfg.clip_num_frames,
            height=cfg.height,
            width=cfg.width,
            start_frames=cfg.start_frames,
            mid_frames=cfg.mid_frames,
            end_frames=cfg.end_frames,
            cut_annotations_path=cfg.cut_annotations_path,
            use_cut_focused_sampling=cfg.use_cut_focused_sampling,
            cut_focus_prob=cfg.cut_focus_prob,
        )

        self.dl = DataLoader(
            dataset, 
            batch_size=cfg.train_batch_size, 
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )

        # Prepare with accelerator
        (
            self.transformer_fwd,
            self.transformer_bwd,
            self.fusion_mlp,
            self.optim_fwd,
            self.optim_bwd,
            self.dl
        ) = self.acc.prepare(
            self.transformer_fwd,
            self.transformer_bwd,
            self.fusion_mlp,
            self.optim_fwd,
            self.optim_bwd,
            self.dl
        )
        
        # Move VAE to device manually
        self.vae = self.vae.to(self.acc.device)
        
        # Cache unwrapped references
        self._unwrapped_transformer_fwd = self.acc.unwrap_model(self.transformer_fwd)
        self._unwrapped_transformer_bwd = self.acc.unwrap_model(self.transformer_bwd)

        # Pre-compute latent normalization tensors
        self.latents_mean = None
        self.latents_std = None

        # Log training info
        if self.acc.is_local_main_process:
            num_gpus = self.acc.num_processes
            effective_batch = cfg.train_batch_size * num_gpus * cfg.gradient_accumulation_steps
            print(f"\n{'='*60}")
            print("BIDIRECTIONAL INBETWEENING TRAINER (Separate LoRAs)")
            print(f"{'='*60}")
            print(f"Training on {num_gpus} GPU(s)")
            print(f"  Per-GPU batch size: {cfg.train_batch_size}")
            print(f"  Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
            print(f"  Effective batch size: {effective_batch}")
            print(f"  Forward LoRA params: {sum(p.numel() for p in params_fwd):,}")
            print(f"  Backward LoRA params: {sum(p.numel() for p in params_bwd):,}")
            fusion_mlp_params = sum(p.numel() for p in self.fusion_mlp.parameters())
            print(f"  Fusion MLP params: {fusion_mlp_params:,} ({'frozen' if self.fusion_mlp_frozen else 'trainable'})")
            print(f"  Alternating every: {cfg.alternating_steps} steps")
            print(f"  Attention: {cfg.attn_implementation}")
            print(f"{'='*60}\n")

        # TensorBoard logging
        self.writer = None
        if self.acc.is_local_main_process:
            log_dir = Path(cfg.log_dir) / "tensorboard_bidirectional"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard logs → {log_dir}")

        # Checkpoint directory
        self.checkpoint_dir = Path(cfg.output_dir) / "checkpoints_bidirectional"
        if self.acc.is_local_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0

    def _sample_timesteps(self, batch_size: int, device: torch.device, power: float = 0.5) -> torch.Tensor:
        """Stratified + power-biased timestep sampling."""
        T = self.scheduler.config.num_train_timesteps
        u = torch.rand(batch_size, device=device)
        strata = (torch.arange(batch_size, device=device, dtype=u.dtype) + u) / batch_size
        if power != 1.0:
            strata = strata ** power
        t = strata * T
        return t

    def save_checkpoint(self, step: int):
        """Save checkpoint with both LoRAs and fusion MLP."""
        if not self.acc.is_local_main_process:
            return

        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save forward LoRA
        unwrapped_fwd = self.acc.unwrap_model(self.transformer_fwd)
        unwrapped_fwd.save_pretrained(checkpoint_path / "lora_fwd")

        # Save backward LoRA
        unwrapped_bwd = self.acc.unwrap_model(self.transformer_bwd)
        unwrapped_bwd.save_pretrained(checkpoint_path / "lora_bwd")

        # Save fusion MLP
        fusion_mlp_unwrapped = self.acc.unwrap_model(self.fusion_mlp)
        torch.save(fusion_mlp_unwrapped.state_dict(), checkpoint_path / "fusion_mlp.pt")

        # Save optimizer states
        torch.save(self.optim_fwd.state_dict(), checkpoint_path / "optimizer_fwd.pt")
        torch.save(self.optim_bwd.state_dict(), checkpoint_path / "optimizer_bwd.pt")

        # Save training state
        state = {
            "global_step": step,
            "config": {
                "learning_rate": self.cfg.learning_rate,
                "num_train_steps": self.cfg.num_train_steps,
                "lora_r": self.cfg.lora_r,
                "lora_alpha": self.cfg.lora_alpha,
                "fusion_hidden_dim": self.cfg.fusion_hidden_dim,
                "alternating_steps": self.cfg.alternating_steps,
            }
        }
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"Checkpoint saved → {checkpoint_path}")
        self._cleanup_old_checkpoints(keep=3)

    def _cleanup_old_checkpoints(self, keep: int = 3):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1])
        )
        for ckpt in checkpoints[:-keep]:
            import shutil
            shutil.rmtree(ckpt)
            print(f"Removed old checkpoint: {ckpt}")

    def load_checkpoint(self, checkpoint_path_str: str):
        """Load checkpoint to resume training."""
        checkpoint_path = Path(checkpoint_path_str)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load training state
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
            self.global_step = state["global_step"]
            print(f"Resuming from step {self.global_step}")

        # Load forward LoRA
        lora_fwd_path = checkpoint_path / "lora_fwd"
        if lora_fwd_path.exists():
            unwrapped_fwd = self.acc.unwrap_model(self.transformer_fwd)
            unwrapped_fwd.load_adapter(str(lora_fwd_path), adapter_name="default")
            print(f"Loaded forward LoRA from {lora_fwd_path}")

        # Load backward LoRA
        lora_bwd_path = checkpoint_path / "lora_bwd"
        if lora_bwd_path.exists():
            unwrapped_bwd = self.acc.unwrap_model(self.transformer_bwd)
            unwrapped_bwd.load_adapter(str(lora_bwd_path), adapter_name="default")
            print(f"Loaded backward LoRA from {lora_bwd_path}")

        # Load fusion MLP
        fusion_path = checkpoint_path / "fusion_mlp.pt"
        if fusion_path.exists():
            fusion_state = torch.load(fusion_path, map_location=self.acc.device)
            fusion_mlp_unwrapped = self.acc.unwrap_model(self.fusion_mlp)
            fusion_mlp_unwrapped.load_state_dict(fusion_state)
            print(f"Loaded fusion MLP from {fusion_path}")

        # Load optimizer states
        optim_fwd_path = checkpoint_path / "optimizer_fwd.pt"
        if optim_fwd_path.exists():
            optim_state = torch.load(optim_fwd_path, map_location=self.acc.device)
            self.optim_fwd.load_state_dict(optim_state)
            print(f"Loaded forward optimizer state from {optim_fwd_path}")

        optim_bwd_path = checkpoint_path / "optimizer_bwd.pt"
        if optim_bwd_path.exists():
            optim_state = torch.load(optim_bwd_path, map_location=self.acc.device)
            self.optim_bwd.load_state_dict(optim_state)
            print(f"Loaded backward optimizer state from {optim_bwd_path}")

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint."""
        if not self.checkpoint_dir.exists():
            return None
        checkpoints = list(self.checkpoint_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: int(p.name.split("-")[1]))

    def train(self):
        """
        Main training loop for bidirectional inbetweening with alternating LoRAs.
        
        Training procedure:
        1. Encode video segments (start, mid, end) to latents
        2. Alternate every N steps between:
           - Forward LoRA: [start, noisy_mid] -> predict mid velocity
           - Backward LoRA: [noisy_mid, end] -> predict mid velocity
        
        Each LoRA is trained separately to avoid gradient conflicts.
        """
        device = self.acc.device
        self.transformer_fwd.train()
        self.transformer_bwd.train()
        self.vae.eval()

        # Resume from checkpoint if specified
        if self.cfg.resume_from_checkpoint:
            if self.cfg.resume_from_checkpoint == "latest":
                latest = self.get_latest_checkpoint()
                if latest:
                    self.load_checkpoint(str(latest))
                else:
                    print("No checkpoint found, starting from scratch")
            else:
                self.load_checkpoint(self.cfg.resume_from_checkpoint)

        step = self.global_step
        remaining_steps = self.cfg.num_train_steps - step
        
        if remaining_steps <= 0:
            print(f"Training already completed ({step}/{self.cfg.num_train_steps} steps)")
            return

        pbar = tqdm(
            total=self.cfg.num_train_steps,
            initial=step,
            disable=not self.acc.is_local_main_process,
            desc="Bidirectional Training"
        )

        # Loss tracking (separate counts for each mode)
        running_losses = {"fwd": 0.0, "bwd": 0.0}
        loss_counts = {"fwd": 0, "bwd": 0}
        
        # Alternating training parameters
        # IMPORTANT: alternating_steps must be a multiple of gradient_accumulation_steps
        # to ensure clean gradient accumulation windows
        alternating_steps = self.cfg.alternating_steps
        grad_acc_steps = self.cfg.gradient_accumulation_steps
        if alternating_steps % grad_acc_steps != 0:
            # Round up to nearest multiple
            alternating_steps = ((alternating_steps // grad_acc_steps) + 1) * grad_acc_steps
            if self.acc.is_local_main_process:
                print(f"⚠️  Adjusted alternating_steps to {alternating_steps} (multiple of grad_acc_steps={grad_acc_steps})")
        
        # Track which phase we're in (forward vs backward) based on completed accumulation cycles
        # Instead of switching every step, we switch after completing alternating_steps worth of steps
        def get_training_mode(s):
            # Each alternating_steps, we switch
            cycle_position = s % (2 * alternating_steps)
            return cycle_position < alternating_steps  # True = forward, False = backward

        while step < self.cfg.num_train_steps:
            for batch in self.dl:
                # Determine which LoRA to train this step
                training_fwd = get_training_mode(step)
                
                # Select the appropriate transformer and optimizer
                if training_fwd:
                    active_transformer = self.transformer_fwd
                    active_optim = self.optim_fwd
                    unwrapped_transformer = self._unwrapped_transformer_fwd
                else:
                    active_transformer = self.transformer_bwd
                    active_optim = self.optim_bwd
                    unwrapped_transformer = self._unwrapped_transformer_bwd
                
                with self.acc.accumulate(active_transformer):
                    # Video: [B, T, C, H, W] -> [B, C, T, H, W]
                    video = batch["video"].to(device, non_blocking=True)
                    video = video.permute(0, 2, 1, 3, 4)
                    B = video.shape[0]

                    # Split video
                    s_frames = self.cfg.start_frames
                    m_frames = self.cfg.mid_frames
                    e_frames = self.cfg.end_frames
                    
                    video_start = video[:, :, :s_frames]
                    video_mid = video[:, :, s_frames:s_frames+m_frames]
                    video_end = video[:, :, s_frames+m_frames:]

                    # Encode segments to latents
                    with torch.no_grad():
                        enc_start = self.vae.encode(video_start)
                        start_lat = retrieve_latents(enc_start)
                        
                        enc_mid = self.vae.encode(video_mid)
                        mid_lat = retrieve_latents(enc_mid)
                        
                        enc_end = self.vae.encode(video_end)
                        end_lat = retrieve_latents(enc_end)
                        
                        # Convert to transformer dtype
                        start_lat = start_lat.to(unwrapped_transformer.dtype)
                        mid_lat = mid_lat.to(unwrapped_transformer.dtype)
                        end_lat = end_lat.to(unwrapped_transformer.dtype)

                        # Normalize latents
                        if self.latents_mean is None:
                            self.latents_mean = torch.tensor(
                                self.vae.config.latents_mean,
                                device=start_lat.device,
                                dtype=start_lat.dtype,
                            ).view(1, self.vae.config.z_dim, 1, 1, 1)
                            self.latents_std = 1.0 / torch.tensor(
                                self.vae.config.latents_std,
                                device=start_lat.device,
                                dtype=start_lat.dtype,
                            ).view(1, self.vae.config.z_dim, 1, 1, 1)
                        
                        start_lat = (start_lat - self.latents_mean) * self.latents_std
                        mid_lat = (mid_lat - self.latents_mean) * self.latents_std
                        end_lat = (end_lat - self.latents_mean) * self.latents_std

                    # Get latent dimensions
                    _, C, T_start_lat, H, W = start_lat.shape
                    _, _, T_mid_lat, _, _ = mid_lat.shape
                    _, _, T_end_lat, _, _ = end_lat.shape
                    
                    # Sample timestep and add noise to mid
                    noise = torch.randn_like(mid_lat)
                    # Use power=0.5 to bias toward cleaner samples (matches original trainer)
                    t = self._sample_timesteps(B, device, power=0.5)
                    t_normalized = t.view(B, 1, 1, 1, 1) / self.scheduler.config.num_train_timesteps
                    noisy_mid = (1 - t_normalized) * mid_lat + t_normalized * noise

                    # Dummy text embeddings
                    enc_state = torch.zeros(
                        B, 1, unwrapped_transformer.config.text_dim,
                        device=device,
                        dtype=unwrapped_transformer.dtype,
                    )

                    # Flow matching target: velocity = noise - x_0
                    target = noise - mid_lat

                    # Initialize loss values for logging
                    loss_fwd_value = 0.0
                    loss_bwd_value = 0.0

                    if training_fwd:
                        # =========================================================
                        # Forward LoRA training: [start, noisy_mid] -> predict mid
                        # =========================================================
                        latents_fwd = torch.cat([start_lat, noisy_mid], dim=2)
                        T_fwd = T_start_lat + T_mid_lat
                        
                        # Mask: start is conditioning, mid is noisy
                        mask_fwd = torch.zeros(B, T_fwd, device=device, dtype=torch.bool)
                        mask_fwd[:, :T_start_lat] = True

                        pred_fwd = active_transformer(
                            hidden_states=latents_fwd,
                            timestep=t,
                            encoder_hidden_states=enc_state,
                            conditioning_mask=mask_fwd,
                            return_dict=True,
                        ).sample

                        # Extract mid prediction: [B, C, T_mid_lat, H, W]
                        pred_mid_fwd = pred_fwd[:, :, T_start_lat:]
                        
                        # Simple MSE loss
                        loss = torch.nn.functional.mse_loss(pred_mid_fwd.float(), target.float())
                        loss_fwd_value = loss.detach().item()
                        
                        self.acc.backward(loss)
                        del latents_fwd, pred_fwd, pred_mid_fwd, loss
                    else:
                        # =========================================================
                        # Backward LoRA training: [noisy_mid, end] -> predict mid
                        # =========================================================
                        latents_bwd = torch.cat([noisy_mid, end_lat], dim=2)
                        T_bwd = T_mid_lat + T_end_lat
                        
                        # Mask: end frames are conditioning
                        mask_bwd = torch.zeros(B, T_bwd, device=device, dtype=torch.bool)
                        mask_bwd[:, T_mid_lat:] = True

                        pred_bwd = active_transformer(
                            hidden_states=latents_bwd,
                            timestep=t,
                            encoder_hidden_states=enc_state,
                            conditioning_mask=mask_bwd,
                            return_dict=True,
                        ).sample

                        # Extract mid prediction: [B, C, T_mid_lat, H, W]
                        pred_mid_bwd = pred_bwd[:, :, :T_mid_lat]
                        
                        # Simple MSE loss
                        loss = torch.nn.functional.mse_loss(pred_mid_bwd.float(), target.float())
                        loss_bwd_value = loss.detach().item()
                        
                        self.acc.backward(loss)
                        del latents_bwd, pred_bwd, pred_mid_bwd, loss

                    # Gradient clipping to prevent exploding gradients
                    if self.acc.sync_gradients:
                        self.acc.clip_grad_norm_(active_transformer.parameters(), max_norm=1.0)
                    
                    # Optimizer step for active LoRA only
                    active_optim.step()
                    active_optim.zero_grad()

                    step += 1
                    pbar.update(1)

                    # Track losses (only update the active mode's counter)
                    if training_fwd:
                        running_losses["fwd"] += loss_fwd_value
                        loss_counts["fwd"] += 1
                    else:
                        running_losses["bwd"] += loss_bwd_value
                        loss_counts["bwd"] += 1

                    # TensorBoard logging
                    if self.writer and step % self.cfg.log_every_n_steps == 0:
                        if self.acc.is_local_main_process:
                            # Compute averages only if we have samples
                            avg_fwd = running_losses["fwd"] / max(loss_counts["fwd"], 1)
                            avg_bwd = running_losses["bwd"] / max(loss_counts["bwd"], 1)
                            
                            # Only log if we have actual samples for that mode
                            if loss_counts["fwd"] > 0:
                                self.writer.add_scalar("train/loss_forward", avg_fwd, step)
                            if loss_counts["bwd"] > 0:
                                self.writer.add_scalar("train/loss_backward", avg_bwd, step)
                            self.writer.add_scalar("train/timestep_mean", t.mean().item(), step)
                            self.writer.add_scalar("train/training_fwd", 1.0 if training_fwd else 0.0, step)
                            
                            mode_str = "FWD" if training_fwd else "BWD"
                            current_loss = loss_fwd_value if training_fwd else loss_bwd_value
                            pbar.set_postfix({
                                "mode": mode_str,
                                "loss": f"{current_loss:.4f}",
                            })
                        
                        running_losses = {k: 0.0 for k in running_losses}
                        loss_counts = {k: 0 for k in loss_counts}

                    # Save checkpoint periodically
                    if step % self.cfg.save_every_n_steps == 0:
                        self.acc.wait_for_everyone()
                        self.save_checkpoint(step)

                    if step >= self.cfg.num_train_steps:
                        break
        
        self.global_step = step
        pbar.close()

        # Final save
        self.acc.wait_for_everyone()
        
        if self.acc.is_local_main_process:
            self.save_checkpoint(step)
            
            # Save final models
            output_path = Path(self.cfg.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.acc.unwrap_model(self.transformer_fwd).save_pretrained(output_path / "lora_fwd")
            self.acc.unwrap_model(self.transformer_bwd).save_pretrained(output_path / "lora_bwd")
            torch.save(
                self.acc.unwrap_model(self.fusion_mlp).state_dict(), 
                output_path / "fusion_mlp.pt"
            )
            print(f"Saved final models → {output_path}")

            if self.writer:
                self.writer.close()
        
        self.acc.wait_for_everyone()
