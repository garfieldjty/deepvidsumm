# scripts/train_fusion_mlp.py
"""
Training script for the Cumulative Softmax Fusion MLP.

This trains the fusion MLP separately by learning to predict fusion weights
from start/end latents, supervised by ground-truth similarity patterns.

The trained MLP can then be used by the bidirectional inbetweening trainer.

Usage:
    python -m src.train_fusion_mlp --config config/fusion_mlp_config.yaml
    
    # With accelerate for multi-GPU
    accelerate launch -m src.train_fusion_mlp --config config/fusion_mlp_config.yaml
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from tqdm.auto import tqdm
from lion_pytorch import Lion

from src.utils import load_config
from src.utils_latents import retrieve_latents
from src.models import load_wan_components
from src.dataset import InbetweenVideoDataset
from src.trainer import CumulativeSoftmaxFusionMLP


@dataclass
class FusionMLPTrainConfig:
    """Configuration for fusion MLP training."""
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
    # VAE precision (transformer not needed for MLP training)
    vae_precision: str = "fp32"
    # Fusion MLP settings
    fusion_hidden_dim: int = 256
    fusion_num_layers: int = 3
    cnn_feature_dim: int = 64  # Output feature dim from CNN spatial encoder
    # Logging
    log_dir: str = "./logs"
    log_every_n_steps: int = 10
    # Checkpointing
    save_every_n_steps: int = 500
    resume_from_checkpoint: Optional[str] = None


class FusionMLPTrainer:
    """
    Trainer for the Cumulative Softmax Fusion MLP.
    
    This trains the MLP to predict fusion weights from start/end latents,
    using ground-truth weights computed from latent similarity.
    """
    
    def __init__(self, cfg: FusionMLPTrainConfig):
        self.cfg = cfg
        
        self.acc = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision="bf16"
        )

        # Load VAE only (we don't need the transformer for MLP training)
        self.vae, _, _ = load_wan_components(
            cfg.base_model_path,
            transformer_precision="bf16",  # doesn't matter, not used
            vae_precision=cfg.vae_precision,
            attn_implementation="sdpa",
        )

        # Get latent channel dimension from VAE config
        latent_dim = self.vae.config.z_dim
        
        # Create fusion MLP with CNN spatial encoder
        self.fusion_mlp = CumulativeSoftmaxFusionMLP(
            latent_dim=latent_dim,
            hidden_dim=cfg.fusion_hidden_dim,
            num_layers=cfg.fusion_num_layers,
            cnn_feature_dim=cfg.cnn_feature_dim,
        )

        # Optimizer
        self.optim = Lion(
            self.fusion_mlp.parameters(), 
            lr=cfg.learning_rate, 
            weight_decay=1e-5
        )

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
        self.fusion_mlp, self.optim, self.dl = self.acc.prepare(
            self.fusion_mlp, self.optim, self.dl
        )
        
        # Move VAE to device manually
        self.vae = self.vae.to(self.acc.device)

        # Pre-compute latent normalization tensors
        self.latents_mean = None
        self.latents_std = None

        # Log training info
        if self.acc.is_local_main_process:
            num_gpus = self.acc.num_processes
            effective_batch = cfg.train_batch_size * num_gpus * cfg.gradient_accumulation_steps
            print(f"\n{'='*60}")
            print("FUSION MLP TRAINER")
            print(f"{'='*60}")
            print(f"Training on {num_gpus} GPU(s)")
            print(f"  Per-GPU batch size: {cfg.train_batch_size}")
            print(f"  Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
            print(f"  Effective batch size: {effective_batch}")
            print(f"  MLP params: {sum(p.numel() for p in self.fusion_mlp.parameters()):,}")
            print(f"  Hidden dim: {cfg.fusion_hidden_dim}")
            print(f"  Num layers: {cfg.fusion_num_layers}")
            print(f"{'='*60}\n")

        # TensorBoard logging
        self.writer = None
        if self.acc.is_local_main_process:
            log_dir = Path(cfg.log_dir) / "tensorboard_fusion_mlp"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard logs → {log_dir}")

        # Checkpoint directory
        self.checkpoint_dir = Path(cfg.output_dir) / "checkpoints_fusion_mlp"
        if self.acc.is_local_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0

    def save_checkpoint(self, step: int):
        """Save checkpoint."""
        if not self.acc.is_local_main_process:
            return

        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save fusion MLP
        fusion_mlp_unwrapped = self.acc.unwrap_model(self.fusion_mlp)
        torch.save(fusion_mlp_unwrapped.state_dict(), checkpoint_path / "fusion_mlp.pt")

        # Save optimizer state
        torch.save(self.optim.state_dict(), checkpoint_path / "optimizer.pt")

        # Save training state
        state = {
            "global_step": step,
            "config": {
                "learning_rate": self.cfg.learning_rate,
                "num_train_steps": self.cfg.num_train_steps,
                "fusion_hidden_dim": self.cfg.fusion_hidden_dim,
                "fusion_num_layers": self.cfg.fusion_num_layers,
            }
        }
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"Checkpoint saved → {checkpoint_path}")
        self._cleanup_old_checkpoints(keep=3)

    def _cleanup_old_checkpoints(self, keep: int = 3):
        """Remove old checkpoints."""
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

        # Load fusion MLP
        fusion_path = checkpoint_path / "fusion_mlp.pt"
        if fusion_path.exists():
            fusion_state = torch.load(fusion_path, map_location=self.acc.device)
            fusion_mlp_unwrapped = self.acc.unwrap_model(self.fusion_mlp)
            fusion_mlp_unwrapped.load_state_dict(fusion_state)
            print(f"Loaded fusion MLP from {fusion_path}")

        # Load optimizer state
        optim_path = checkpoint_path / "optimizer.pt"
        if optim_path.exists():
            optim_state = torch.load(optim_path, map_location=self.acc.device)
            self.optim.load_state_dict(optim_state)
            print(f"Loaded optimizer state from {optim_path}")

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
        Main training loop for fusion MLP.
        
        Training procedure:
        1. Encode video segments (start, mid, end) to latents
        2. Predict fusion weights from start/end latents via MLP
        3. Compute GT weights from latent similarity
        4. MSE loss between predicted and GT weights
        """
        device = self.acc.device
        self.fusion_mlp.train()
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
            desc="Fusion MLP Training"
        )

        # Loss tracking
        running_loss = 0.0
        loss_count = 0

        while step < self.cfg.num_train_steps:
            for batch in self.dl:
                with self.acc.accumulate(self.fusion_mlp):
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
                    _, _, T_mid_lat, _, _ = mid_lat.shape

                    # Predict fusion weights
                    w_fwd_pred, w_bwd_pred = self.fusion_mlp(start_lat, end_lat, T_mid_lat)
                    
                    # Compute GT weights from latent similarity
                    with torch.no_grad():
                        w_fwd_gt, w_bwd_gt = CumulativeSoftmaxFusionMLP.compute_gt_weights_from_similarity(
                            mid_lat, start_lat, end_lat
                        )
                    
                    # MSE loss
                    loss = torch.nn.functional.mse_loss(w_fwd_pred, w_fwd_gt)

                    self.acc.backward(loss)
                    self.optim.step()
                    self.optim.zero_grad()

                    step += 1
                    pbar.update(1)

                    # Track loss
                    running_loss += loss.detach().item()
                    loss_count += 1

                    # TensorBoard logging
                    if self.writer and step % self.cfg.log_every_n_steps == 0:
                        if self.acc.is_local_main_process:
                            avg_loss = running_loss / loss_count
                            
                            self.writer.add_scalar("train/loss", avg_loss, step)
                            
                            # Log weight statistics
                            w_fwd_mean = w_fwd_pred.mean().item()
                            w_bwd_mean = w_bwd_pred.mean().item()
                            w_fwd_gt_mean = w_fwd_gt.mean().item()
                            
                            self.writer.add_scalar("train/w_fwd_pred_mean", w_fwd_mean, step)
                            self.writer.add_scalar("train/w_bwd_pred_mean", w_bwd_mean, step)
                            self.writer.add_scalar("train/w_fwd_gt_mean", w_fwd_gt_mean, step)
                            
                            pbar.set_postfix({"loss": f"{avg_loss:.6f}"})
                        
                        running_loss = 0.0
                        loss_count = 0

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
            
            # Save final model to output_dir
            output_path = Path(self.cfg.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            torch.save(
                self.acc.unwrap_model(self.fusion_mlp).state_dict(), 
                output_path / "fusion_mlp.pt"
            )
            
            # Also save config for reference
            config_info = {
                "fusion_hidden_dim": self.cfg.fusion_hidden_dim,
                "fusion_num_layers": self.cfg.fusion_num_layers,
                "latent_dim": self.vae.config.z_dim,
                "num_train_steps": self.cfg.num_train_steps,
            }
            with open(output_path / "fusion_mlp_config.json", "w") as f:
                json.dump(config_info, f, indent=2)
            
            print(f"Saved final fusion MLP → {output_path / 'fusion_mlp.pt'}")

            if self.writer:
                self.writer.close()
        
        self.acc.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(
        description="Train Fusion MLP for bidirectional video inbetweening"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/fusion_mlp_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint. Use 'latest' for most recent, or provide path.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # Override resume_from_checkpoint if provided via CLI
    if args.resume:
        cfg["resume_from_checkpoint"] = args.resume

    # Map YAML -> FusionMLPTrainConfig
    train_cfg = FusionMLPTrainConfig(
        base_model_path=cfg["base_model_path"],
        data_root=cfg["data_root"],
        video_glob=cfg.get("video_glob", "**/*.mp4"),
        clip_num_frames=cfg["clip_num_frames"],
        height=cfg["resolution"]["height"],
        width=cfg["resolution"]["width"],
        start_frames=cfg["start_frames"],
        mid_frames=cfg["mid_frames"],
        end_frames=cfg["end_frames"],
        cut_annotations_path=cfg.get("cut_annotations_path", None),
        use_cut_focused_sampling=cfg.get("use_cut_focused_sampling", True),
        cut_focus_prob=cfg.get("cut_focus_prob", 0.7),
        train_batch_size=cfg["train_batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        num_train_steps=cfg["num_train_steps"],
        learning_rate=cfg.get("learning_rate", 1e-4),
        seed=cfg.get("seed", 42),
        output_dir=cfg["output_dir"],
        vae_precision=cfg.get("vae_precision", "fp32"),
        fusion_hidden_dim=cfg.get("fusion_hidden_dim", 256),
        fusion_num_layers=cfg.get("fusion_num_layers", 3),
        cnn_feature_dim=cfg.get("cnn_feature_dim", 64),
        log_dir=cfg.get("log_dir", "./logs"),
        log_every_n_steps=cfg.get("log_every_n_steps", 10),
        save_every_n_steps=cfg.get("save_every_n_steps", 500),
        resume_from_checkpoint=cfg.get("resume_from_checkpoint", None),
    )

    print("\n" + "=" * 60)
    print("FUSION MLP TRAINING")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Output: {train_cfg.output_dir}")
    print(f"Hidden dim: {train_cfg.fusion_hidden_dim}")
    print(f"Num layers: {train_cfg.fusion_num_layers}")
    print(f"CNN feature dim: {train_cfg.cnn_feature_dim}")
    print("=" * 60 + "\n")

    trainer = FusionMLPTrainer(train_cfg)
    trainer.train()


if __name__ == "__main__":
    main()