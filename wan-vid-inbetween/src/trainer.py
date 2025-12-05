from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
import json
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from tqdm.auto import tqdm
from lion_pytorch import Lion

from .utils_latents import retrieve_latents
from .models import load_wan_components, add_lora_to_transformer
from .dataset import InbetweenVideoDataset

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
        self.optim = Lion(params, lr=cfg.learning_rate, weight_decay=1e-3)

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
