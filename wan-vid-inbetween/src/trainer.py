from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

from .utils_latents import retrieve_latents
from .models import load_wan_components, add_lora_to_transformer
from .dataset import InbetweenVideoDataset


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


class InbetweenTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.acc = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps
        )

        # Load components (custom Wan transformer with conditioning mask)
        self.vae, self.transformer, self.scheduler = load_wan_components(
            cfg.base_model_path,
            transformer_precision=cfg.transformer_precision,
            vae_precision=cfg.vae_precision
        )

        # Add LoRA
        self.transformer = add_lora_to_transformer(
            self.transformer,
            cfg.lora_r,
            cfg.lora_alpha,
            cfg.lora_dropout
        )

        # Freeze all except LoRA
        for name, p in self.transformer.named_parameters():
            p.requires_grad = ("lora_" in name)

        params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optim = torch.optim.AdamW(params, lr=cfg.learning_rate)

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

        self.dl = DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True)

        self.vae, self.transformer, self.optim, self.dl = self.acc.prepare(
            self.vae, self.transformer, self.optim, self.dl
        )

    # ------------------------------------------------------------
    # TRAINING STEP (mask-based [start, mid, end] conditioning)
    # ------------------------------------------------------------
    def train(self):
        device = self.acc.device
        self.transformer.train()
        self.vae.eval()

        pbar = tqdm(range(self.cfg.num_train_steps), disable=not self.acc.is_local_main_process)

        step = 0
        while step < self.cfg.num_train_steps:
            for batch in self.dl:
                with self.acc.accumulate(self.transformer):

                    # Video: [B, T, C, H, W] -> [B, C, T, H, W]
                    video = batch["video"].to(device)
                    video = video.permute(0, 2, 1, 3, 4)

                    # Encode via VAE
                    with torch.no_grad():
                        enc = self.vae.encode(video)
                        latents = retrieve_latents(enc)  # [B, C, T_lat, H_lat, W_lat]
                        latents = latents.to(self.transformer.dtype)

                        # Normalize latents using Wan's config (same as inference)
                        latents_mean = torch.tensor(
                            self.vae.config.latents_mean,
                            device=latents.device,
                            dtype=latents.dtype,
                        ).view(1, self.vae.config.z_dim, 1, 1, 1)
                        latents_std = 1.0 / torch.tensor(
                            self.vae.config.latents_std,
                            device=latents.device,
                            dtype=latents.dtype,
                        ).view(1, self.vae.config.z_dim, 1, 1, 1)
                        latents = (latents - latents_mean) * latents_std

                    # Split into start/mid/end (proportional in latent time)
                    B, C, T_lat, H, W = latents.shape
                    s = int(T_lat * self.cfg.start_frames / self.cfg.clip_num_frames)
                    m = int(T_lat * self.cfg.mid_frames / self.cfg.clip_num_frames)
                    e = T_lat - s - m
                    assert s > 0 and m > 0 and e > 0, "Invalid latent splits"

                    # latent order is already [start, mid, end]
                    start_lat = latents[:, :, :s]
                    mid_lat   = latents[:, :, s:s+m]
                    end_lat   = latents[:, :, s+m:]

                    # Noise & timestep (only for mid)
                    noise = torch.randn_like(mid_lat)
                    t = torch.randint(
                        0, self.scheduler.config.num_train_timesteps, (B,), device=device
                    )

                    noisy_mid = self.scheduler.scale_noise(mid_lat, t, noise)

                    # Build transformer input: [start, noisy_mid, end]
                    latents_in = latents.clone()
                    latents_in[:, :, s:s+m] = noisy_mid

                    # Frame-level conditioning mask: True for start+end (conditioning), False for mid
                    conditioning_mask = torch.zeros(
                        B, T_lat, device=device, dtype=torch.bool
                    )
                    conditioning_mask[:, :s] = True
                    conditioning_mask[:, s+m:] = True

                    # Dummy text embeddings (unconditional)
                    enc_state = torch.zeros(
                        B, 1, self.transformer.config.text_dim,
                        device=device,
                        dtype=self.transformer.dtype,
                    )

                    # Forward
                    pred = self.transformer(
                        hidden_states=latents_in,
                        timestep=t,
                        encoder_hidden_states=enc_state,
                        conditioning_mask=conditioning_mask,
                        return_dict=True,
                    ).sample  # [B, C, T_lat, H, W]

                    # Only supervise mid region
                    pred_mid = pred[:, :, s:s+m]
                    loss = torch.nn.functional.mse_loss(pred_mid.float(), noise.float())

                    self.acc.backward(loss)
                    self.optim.step()
                    self.optim.zero_grad()

                    step += 1
                    pbar.update(1)

                    if step >= self.cfg.num_train_steps:
                        break

        if self.acc.is_local_main_process:
            self.acc.unwrap_model(self.transformer).save_pretrained(self.cfg.output_dir)
            print("Saved LoRA →", self.cfg.output_dir)
