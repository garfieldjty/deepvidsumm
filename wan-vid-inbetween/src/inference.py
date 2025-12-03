# src/inference.py

from typing import List
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils import export_to_video

from peft import PeftModel

from .wan_condition_transformer import WanTransformer3DModel
from .utils_latents import retrieve_latents


def load_wan_with_lora(
    base_model_path: str,
    lora_path: str,
    transformer_precision: str = "bf16",
    vae_precision: str = "fp32",
):
    dtype_t = torch.bfloat16 if transformer_precision == "bf16" else torch.float16
    dtype_vae = torch.float32 if vae_precision == "fp32" else torch.float16

    vae = AutoencoderKLWan.from_pretrained(
        base_model_path, subfolder="vae", torch_dtype=dtype_vae
    )
    transformer = WanTransformer3DModel.from_pretrained(
        base_model_path, subfolder="transformer", torch_dtype=dtype_t
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model_path, subfolder="scheduler"
    )

    # Attach LoRA adapter
    transformer = PeftModel.from_pretrained(transformer, lora_path)
    transformer.to(dtype_t)

    return vae, transformer, scheduler


def _read_frames_around_cut(
    video_path: str,
    cut_frame_index: int,
    start_frames: int,
    end_frames: int,
) -> List[Image.Image]:
    """
    Load exactly start_frames before cut and end_frames after cut.

    cut_frame_index is assumed to be the index of the FIRST frame
    of the second shot (0-based). So:
        pre-cut frames: [c - start_frames, ..., c - 1]
        post-cut frames: [c, ..., c + end_frames - 1]
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video {video_path} has 0 frames")

    # Clamp
    cut_frame_index = max(0, min(cut_frame_index, total - 1))

    start_idx = max(0, cut_frame_index - start_frames)
    end_idx = min(total, cut_frame_index + end_frames)

    wanted_idxs = list(range(start_idx, cut_frame_index)) + list(
        range(cut_frame_index, end_idx)
    )

    frames = {}
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in wanted_idxs:
            frames[frame_id] = frame[..., ::-1]  # BGR->RGB
        frame_id += 1
        if frame_id > wanted_idxs[-1]:
            break

    cap.release()

    ordered = [frames[i] for i in wanted_idxs]
    pil_frames = [Image.fromarray(f) for f in ordered]

    # If not enough frames, pad last
    if len(pil_frames) < (start_frames + end_frames):
        last = pil_frames[-1]
        pil_frames += [last] * ((start_frames + end_frames) - len(pil_frames))

    return pil_frames


def _pil_to_tensor_video(
    frames: List[Image.Image],
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Frames -> [1, 3, T, H, W] in [0,1]
    """
    arrs = []
    for f in frames:
        f = f.resize((width, height), Image.BICUBIC)
        a = np.array(f).astype(np.float32) / 255.0  # [H, W, 3]
        a = np.transpose(a, (2, 0, 1))  # [3, H, W]
        arrs.append(a)

    video_np = np.stack(arrs, axis=0)  # [T, 3, H, W]
    video_np = np.transpose(video_np, (1, 0, 2, 3))  # [3, T, H, W]
    video = torch.from_numpy(video_np).unsqueeze(0)  # [1, 3, T, H, W]
    return video


def generate_inbetween_around_cut(
    base_model_path: str,
    lora_path: str,
    video_path: str,
    cut_frame_index: int,
    start_frames: int,
    mid_frames: int,
    end_frames: int,
    height: int,
    width: int,
    num_inference_steps: int = 50,
    out_fps: int = 24,
    transformer_precision: str = "bf16",
    vae_precision: str = "fp32",
    output_path: str = "inbetween_output.mp4",
):
    """
    Use start_frames before the cut + end_frames after the cut as conditioning.
    Generate mid_frames inbetween purely from the model.

    Transformer sees latents in order [start, mid, end] with a mask marking
    start+end as conditioning tokens.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load components + LoRA
    vae, transformer, scheduler = load_wan_with_lora(
        base_model_path=base_model_path,
        lora_path=lora_path,
        transformer_precision=transformer_precision,
        vae_precision=vae_precision,
    )
    vae.to(device)
    transformer.to(device)

    # 2) Load conditioning frames around the cut
    cond_frames = _read_frames_around_cut(
        video_path=video_path,
        cut_frame_index=cut_frame_index,
        start_frames=start_frames,
        end_frames=end_frames,
    )  # list of PIL, len = start_frames + end_frames

    # 3) Convert to tensor for VAE
    video_cond = _pil_to_tensor_video(cond_frames, height, width).to(device)  # [1, 3, T_cond_raw, H, W]

    # 4) Encode to normalized latents
    with torch.no_grad():
        enc = vae.encode(video_cond)
        latents = retrieve_latents(enc)  # [1, C, T_lat_cond, H_lat, W_lat]

        latents_mean = torch.tensor(
            vae.config.latents_mean, device=latents.device, dtype=latents.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            vae.config.latents_std, device=latents.device, dtype=latents.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        cond_latents = (latents - latents_mean) * latents_std

    B, C, T_cond_lat, H_lat, W_lat = cond_latents.shape

    # 5) Split cond latents into start / end according to raw frame ratio
    total_cond_frames = start_frames + end_frames
    frac_start = start_frames / float(total_cond_frames)
    T_start_lat = int(round(T_cond_lat * frac_start))
    T_start_lat = max(1, min(T_start_lat, T_cond_lat - 1))
    T_end_lat = T_cond_lat - T_start_lat

    start_lat = cond_latents[:, :, :T_start_lat]
    end_lat = cond_latents[:, :, T_start_lat:]

    # 6) Choose number of latent frames for mid
    # Training used cond:mid ≈ 1:1, so we mirror that here
    T_mid_lat = T_cond_lat

    # 7) Initialize mid latents as noise
    latents_mid = torch.randn(
        B, C, T_mid_lat, H_lat, W_lat,
        device=device,
        dtype=transformer.dtype,
    )

    # 8) Build frame-level conditioning mask over latent time [start, mid, end]
    T_total_lat = T_start_lat + T_mid_lat + T_end_lat
    conditioning_mask = torch.zeros(B, T_total_lat, device=device, dtype=torch.bool)
    conditioning_mask[:, :T_start_lat] = True
    conditioning_mask[:, T_start_lat + T_mid_lat:] = True  # end segment

    # 9) Scheduler timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # 10) Dummy text encoder hidden states (unconditional)
    text_dim = transformer.config.text_dim
    encoder_hidden_states = torch.zeros(
        B, 1, text_dim, device=device, dtype=transformer.dtype
    )

    # 11) Denoising loop: update mid latents only, cond is fixed
    for t in timesteps:
        # Build [start, mid, end] at current step
        model_input = torch.cat([start_lat, latents_mid, end_lat], dim=2)  # [B, C, T_total_lat, H_lat, W_lat]

        with torch.no_grad():
            out = transformer(
                hidden_states=model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                conditioning_mask=conditioning_mask,
                return_dict=True,
            ).sample  # [B, C, T_total_lat, H_lat, W_lat]

        # Extract mid prediction
        model_output_mid = out[:, :, T_start_lat:T_start_lat + T_mid_lat]

        # Scheduler step for mid only
        step_out = scheduler.step(
            model_output=model_output_mid,
            timestep=t,
            sample=latents_mid,
        )
        latents_mid = step_out.prev_sample

    # 12) Decode mid latents back to RGB frames
    latents_mean = torch.tensor(
        vae.config.latents_mean, device=latents_mid.device, dtype=latents_mid.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=latents_mid.device, dtype=latents_mid.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_mid_unnorm = latents_mid / latents_std + latents_mean

    with torch.no_grad():
        dec = vae.decode(latents_mid_unnorm).sample  # [1, 3, T_mid_lat, H, W]
        dec = dec.clamp(-1, 1)
        dec = (dec + 1.0) / 2.0  # [-1,1] -> [0,1]
        dec_np = dec.squeeze(0).cpu().numpy()  # [3, T_mid_lat, H, W]
        dec_np = np.transpose(dec_np, (1, 2, 3, 0))  # [T_mid_lat, H, W, 3]
        dec_np = (dec_np * 255.0).astype(np.uint8)

    mid_frames_out = [Image.fromarray(frame) for frame in dec_np]

    # 13) Build final sequence: original start + generated mid + original end
    final_frames = cond_frames[:start_frames] + mid_frames_out + cond_frames[-end_frames:]
    export_to_video(final_frames, output_path, fps=out_fps)

    return output_path
