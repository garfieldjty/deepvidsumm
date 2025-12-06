# src/inference.py

import os
from typing import List, Optional
from pathlib import Path

import torch
import torch.nn as nn
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
from .dataset import read_video_frames, resize_frames
from .trainer import CumulativeSoftmaxFusionNet


def load_wan_with_lora(
    base_model_path: str,
    lora_path: str,
    transformer_precision: str = "bf16",
    vae_precision: str = "fp32",
    attn_implementation: str = "sdpa",
):
    dtype_t = torch.bfloat16 if transformer_precision == "bf16" else torch.float16
    dtype_vae = torch.float32 if vae_precision == "fp32" else torch.float16

    vae = AutoencoderKLWan.from_pretrained(
        base_model_path, subfolder="vae", torch_dtype=dtype_vae
    )
    transformer = WanTransformer3DModel.from_pretrained(
        base_model_path, subfolder="transformer", torch_dtype=dtype_t,
        attn_implementation=attn_implementation,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model_path, subfolder="scheduler"
    )

    # Attach LoRA adapter
    transformer = PeftModel.from_pretrained(transformer, lora_path)
    transformer.to(dtype_t)

    return vae, transformer, scheduler


def load_wan_bidirectional(
    base_model_path: str,
    lora_fwd_path: str,
    lora_bwd_path: str,
    fusion_net_path: str,
    transformer_precision: str = "bf16",
    vae_precision: str = "fp32",
    attn_implementation: str = "sdpa",
    fusion_hidden_dim: int = 256,
    fusion_num_layers: int = 3,
    cnn_feature_dim: int = 64,
):
    """
    Load Wan model components for bidirectional inference (memory-efficient version).
    
    This loads only ONE transformer and swaps LoRAs during inference to save memory.
    The LoRA paths are returned for later loading/unloading during the denoising loop.
    
    Args:
        base_model_path: Path to base Wan model
        lora_fwd_path: Path to forward LoRA weights
        lora_bwd_path: Path to backward LoRA weights
        fusion_net_path: Path to pre-trained fusion network weights
        transformer_precision: Precision for transformer
        vae_precision: Precision for VAE
        attn_implementation: Attention implementation
        fusion_hidden_dim: Hidden dim of fusion network (must match training)
        fusion_num_layers: Num layers in fusion network (must match training)
        cnn_feature_dim: CNN feature dim in fusion network (must match training)
    
    Returns:
        vae, transformer (base, no LoRA), scheduler, fusion_net, lora_fwd_path, lora_bwd_path
    """
    dtype_t = torch.bfloat16 if transformer_precision == "bf16" else torch.float16
    dtype_vae = torch.float32 if vae_precision == "fp32" else torch.float16

    vae = AutoencoderKLWan.from_pretrained(
        base_model_path, subfolder="vae", torch_dtype=dtype_vae
    )
    
    # Load base transformer without any LoRA (we'll swap LoRAs during inference)
    transformer = WanTransformer3DModel.from_pretrained(
        base_model_path, subfolder="transformer", torch_dtype=dtype_t,
        attn_implementation=attn_implementation,
    )
    transformer.eval()
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model_path, subfolder="scheduler"
    )
    
    # Load fusion network
    latent_dim = vae.config.z_dim
    fusion_net = CumulativeSoftmaxFusionNet(
        latent_dim=latent_dim,
        hidden_dim=fusion_hidden_dim,
        num_layers=fusion_num_layers,
        cnn_feature_dim=cnn_feature_dim,
    )
    fusion_net.load_state_dict(torch.load(fusion_net_path, map_location="cpu"))
    fusion_net.eval()
    fusion_net.to(dtype_t)

    return vae, transformer, scheduler, fusion_net, lora_fwd_path, lora_bwd_path


def _frames_to_tensor(frames: List[np.ndarray], height: int, width: int) -> torch.Tensor:
    """
    Convert frames from training format to tensor for VAE encoding.
    
    Args:
        frames: list of numpy arrays [H, W, 3] in uint8 RGB
        height: target height for resizing
        width: target width for resizing
    
    Returns:
        tensor [1, 3, T, H, W] in [-1, 1] range (as expected by Wan VAE)
    """
    # Resize frames using training's resize function
    resized = resize_frames(frames, height, width)
    
    # Stack and convert to float32, then normalize to [-1, 1] (Wan VAE expects this range)
    video_np = np.stack(resized, axis=0)  # [T, H, W, 3]
    video_np = video_np.astype(np.float32) * (2.0 / 255.0) - 1.0
    
    # Transpose to [T, 3, H, W]
    video_np = np.transpose(video_np, (0, 3, 1, 2))
    
    # Then to [3, T, H, W]
    video_np = np.transpose(video_np, (1, 0, 2, 3))
    
    # Add batch dimension [1, 3, T, H, W]
    video = torch.from_numpy(video_np).unsqueeze(0)
    
    return video


def _read_frames_around_cut(
    video_path: str,
    cut_frame_index: int,
    start_frames: int,
    end_frames: int,
) -> List[np.ndarray]:
    """
    Load exactly start_frames before cut and end_frames after cut.
    Uses training's read_video_frames function.

    cut_frame_index is assumed to be the index of the FIRST frame
    of the second shot (0-based). So:
        pre-cut frames: [c - start_frames, ..., c - 1]
        post-cut frames: [c, ..., c + end_frames - 1]
        
    Returns:
        list of numpy arrays [H, W, 3] in uint8 RGB
    """
    # Read pre-cut frames
    start_idx = max(0, cut_frame_index - start_frames)
    pre_frames = read_video_frames(
        video_path, 
        num_frames=start_frames,
        start_index=start_idx,
    )
    
    # Read post-cut frames
    post_frames = read_video_frames(
        video_path,
        num_frames=end_frames,
        start_index=cut_frame_index,
    )
    
    # Concatenate the lists
    frames = pre_frames + post_frames
    
    return frames


def _read_frames_from_video(
    video_path: str,
    start_frame_index: int,
    duration: int,
) -> List[np.ndarray]:
    """
    Load exactly 'duration' frames from video starting at start_frame_index.
    Uses training's read_video_frames function.
    
    Args:
        video_path: Path to the video file
        start_frame_index: 0-based index of the first frame to read
        duration: Number of frames to read
    
    Returns:
        list of numpy arrays [H, W, 3] in uint8 RGB
    """
    frames = read_video_frames(
        video_path,
        num_frames=duration,
        start_index=start_frame_index,
    )
    
    return frames


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

    # 2) Load conditioning frames around the cut using training's function
    cond_frames = _read_frames_around_cut(
        video_path=video_path,
        cut_frame_index=cut_frame_index,
        start_frames=start_frames,
        end_frames=end_frames,
    )  # list of np.ndarray [H, W, 3]

    # 3) Encode start and end frames SEPARATELY (matches training)
    start_frames_list = cond_frames[:start_frames]
    end_frames_list = cond_frames[start_frames:]
    
    video_start = _frames_to_tensor(start_frames_list, height, width).to(device)
    video_end = _frames_to_tensor(end_frames_list, height, width).to(device)

    # 4) Encode start and end to normalized latents separately
    with torch.no_grad():
        # Encode start
        enc_start = vae.encode(video_start)
        latents_start = retrieve_latents(enc_start)
        
        # Encode end
        enc_end = vae.encode(video_end)
        latents_end = retrieve_latents(enc_end)

        # Normalize both using Wan's config
        latents_mean = torch.tensor(
            vae.config.latents_mean, device=latents_start.device, dtype=latents_start.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            vae.config.latents_std, device=latents_start.device, dtype=latents_start.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        
        start_lat = (latents_start - latents_mean) * latents_std
        end_lat = (latents_end - latents_mean) * latents_std
        
        # Convert to transformer dtype for consistency
        start_lat = start_lat.to(dtype=transformer.dtype)
        end_lat = end_lat.to(dtype=transformer.dtype)

    B, C, T_start_lat, H_lat, W_lat = start_lat.shape
    _, _, T_end_lat, _, _ = end_lat.shape

    # 5) Calculate mid latent frames proportionally to match training
    # Training splits: T_mid_lat such that start:mid:end ratio matches frame ratio
    total_frames = start_frames + mid_frames + end_frames
    T_mid_lat = int(round((T_start_lat + T_end_lat) * mid_frames / (start_frames + end_frames)))

    # 6) Initialize mid latents as noise
    latents_mid = torch.randn(
        B, C, T_mid_lat, H_lat, W_lat,
        device=device,
        dtype=transformer.dtype,
    )

    # 7) Build frame-level conditioning mask over latent time [start, mid, end]
    T_total_lat = T_start_lat + T_mid_lat + T_end_lat
    conditioning_mask = torch.zeros(B, T_total_lat, device=device, dtype=torch.bool)
    conditioning_mask[:, :T_start_lat] = True
    conditioning_mask[:, T_start_lat + T_mid_lat:] = True  # end segment

    # 8) Scheduler timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # 9) Dummy text encoder hidden states (unconditional)
    text_dim = transformer.config.text_dim
    encoder_hidden_states = torch.zeros(
        B, 1, text_dim, device=device, dtype=transformer.dtype
    )

    # 10) Denoising loop: update mid latents only, cond is fixed
    for t in timesteps:
        # Build [start, mid, end] at current step
        model_input = torch.cat([start_lat, latents_mid, end_lat], dim=2)  # [B, C, T_total_lat, H_lat, W_lat]
        
        # Ensure timestep is properly shaped as 1D tensor
        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                timestep = t.unsqueeze(0)  # Convert scalar to 1D tensor [1]
            else:
                timestep = t.view(-1)  # Flatten to 1D
        else:
            timestep = torch.tensor([t], device=device)  # Convert scalar to 1D tensor

        with torch.no_grad():
            out = transformer(
                hidden_states=model_input,
                timestep=timestep,
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

    # 11) MATCH TRAINING: Decode the FULL sequence [start, mid, end] together
    # Concatenate all three segments
    latents_full = torch.cat([start_lat, latents_mid, end_lat], dim=2)  # [B, C, T_total_lat, H_lat, W_lat]
    
    # Unnormalize the full latent sequence
    latents_mean = torch.tensor(
        vae.config.latents_mean, device=latents_full.device, dtype=latents_full.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=latents_full.device, dtype=latents_full.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_full_unnorm = latents_full / latents_std + latents_mean
    
    # Convert back to VAE dtype before decoding
    latents_full_unnorm = latents_full_unnorm.to(dtype=vae.dtype)

    with torch.no_grad():
        # Decode the entire sequence at once (matches training)
        dec = vae.decode(latents_full_unnorm).sample  # [1, 3, T_total_lat, H, W]
        # VAE outputs in [-1, 1] range, convert to [0, 255] for uint8
        dec = dec.clamp(-1, 1)
        dec_np = dec.squeeze(0).cpu().numpy()  # [3, T_total_lat, H, W]
        dec_np = np.transpose(dec_np, (1, 2, 3, 0))  # [T_total_lat, H, W, 3]
        # Convert from [-1, 1] to [0, 255]
        dec_np = ((dec_np / 2.0 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

    # 12) Convert all decoded frames to PIL
    all_frames = [Image.fromarray(frame) for frame in dec_np]
    
    # Calculate frame splits (proportional to original frame counts)
    # Training's split logic: frames are proportionally distributed
    total_decoded_frames = len(all_frames)
    frac_start = start_frames / float(total_frames)
    frac_mid = mid_frames / float(total_frames)
    
    n_start_decoded = int(round(total_decoded_frames * frac_start))
    n_mid_decoded = int(round(total_decoded_frames * frac_mid))
    n_end_decoded = total_decoded_frames - n_start_decoded - n_mid_decoded
    
    # Ensure we have at least 1 frame in each segment
    n_start_decoded = max(1, n_start_decoded)
    n_mid_decoded = max(1, n_mid_decoded)
    n_end_decoded = max(1, total_decoded_frames - n_start_decoded - n_mid_decoded)
    
    # Extract the segments
    start_decoded = all_frames[:n_start_decoded]
    mid_decoded = all_frames[n_start_decoded:n_start_decoded + n_mid_decoded]
    end_decoded = all_frames[n_start_decoded + n_mid_decoded:]
    
    # 13) Build final output video with all frames
    final_frames = start_decoded + mid_decoded + end_decoded
    export_to_video(final_frames, output_path, fps=out_fps)

    return output_path




def generate_inbetween_from_two_videos(
    base_model_path: str,
    lora_path: str,
    start_video_path: str,
    start_frame_index: int,
    start_duration: int,
    end_video_path: str,
    end_frame_index: int,
    end_duration: int,
    mid_frames: int,
    height: int,
    width: int,
    num_inference_steps: int = 50,
    out_fps: int = 24,
    transformer_precision: str = "bf16",
    vae_precision: str = "fp32",
    output_path: str = "inbetween_output.mp4",
):
    """
    Generate inbetween frames between two video segments.
    
    Args:
        base_model_path: Path to base Wan model
        lora_path: Path to trained LoRA weights
        start_video_path: Path to first video
        start_frame_index: Starting frame index in first video
        start_duration: Number of frames to use from first video
        end_video_path: Path to second video
        end_frame_index: Starting frame index in second video
        end_duration: Number of frames to use from second video
        mid_frames: Number of frames to generate between the two segments
        height: Output height
        width: Output width
        num_inference_steps: Number of denoising steps
        out_fps: Output video FPS
        transformer_precision: Precision for transformer ("bf16" or "fp16")
        vae_precision: Precision for VAE ("fp32" or "fp16")
        output_path: Path to save the output video
    
    Returns:
        Path to the output video
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

    # 2) Load frames from both videos using training's function
    start_frames_list = _read_frames_from_video(
        start_video_path, start_frame_index, start_duration
    )
    end_frames_list = _read_frames_from_video(
        end_video_path, end_frame_index, end_duration
    )

    # 3) Encode start and end frames SEPARATELY (matches training)
    video_start = _frames_to_tensor(start_frames_list, height, width).to(device)
    video_end = _frames_to_tensor(end_frames_list, height, width).to(device)

    # 4) Encode start and end to normalized latents separately
    with torch.no_grad():
        # Encode start
        enc_start = vae.encode(video_start)
        latents_start = retrieve_latents(enc_start)
        
        # Encode end
        enc_end = vae.encode(video_end)
        latents_end = retrieve_latents(enc_end)

        # Normalize both using Wan's config
        latents_mean = torch.tensor(
            vae.config.latents_mean, device=latents_start.device, dtype=latents_start.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            vae.config.latents_std, device=latents_start.device, dtype=latents_start.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        
        start_lat = (latents_start - latents_mean) * latents_std
        end_lat = (latents_end - latents_mean) * latents_std
        
        # Convert to transformer dtype for consistency
        start_lat = start_lat.to(dtype=transformer.dtype)
        end_lat = end_lat.to(dtype=transformer.dtype)

    B, C, T_start_lat, H_lat, W_lat = start_lat.shape
    _, _, T_end_lat, _, _ = end_lat.shape

    # 5) Calculate mid latent frames proportionally to match training
    total_frames = start_duration + mid_frames + end_duration
    T_mid_lat = int(round((T_start_lat + T_end_lat) * mid_frames / (start_duration + end_duration)))

    # 6) Initialize mid latents as noise
    latents_mid = torch.randn(
        B, C, T_mid_lat, H_lat, W_lat,
        device=device,
        dtype=transformer.dtype,
    )

    # 7) Build frame-level conditioning mask
    T_total_lat = T_start_lat + T_mid_lat + T_end_lat
    conditioning_mask = torch.zeros(B, T_total_lat, device=device, dtype=torch.bool)
    conditioning_mask[:, :T_start_lat] = True
    conditioning_mask[:, T_start_lat + T_mid_lat:] = True

    # 8) Scheduler timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # 9) Dummy text encoder hidden states
    text_dim = transformer.config.text_dim
    encoder_hidden_states = torch.zeros(
        B, 1, text_dim, device=device, dtype=transformer.dtype
    )

    # 10) Denoising loop
    for t in timesteps:
        model_input = torch.cat([start_lat, latents_mid, end_lat], dim=2)
        
        # Ensure timestep is properly shaped as 1D tensor
        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                timestep = t.unsqueeze(0)  # Convert scalar to 1D tensor [1]
            else:
                timestep = t.view(-1)  # Flatten to 1D
        else:
            timestep = torch.tensor([t], device=device)  # Convert scalar to 1D tensor

        with torch.no_grad():
            out = transformer(
                hidden_states=model_input,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                conditioning_mask=conditioning_mask,
                return_dict=True,
            ).sample

        model_output_mid = out[:, :, T_start_lat:T_start_lat + T_mid_lat]

        step_out = scheduler.step(
            model_output=model_output_mid,
            timestep=t,
            sample=latents_mid,
        )
        latents_mid = step_out.prev_sample

    # 11) MATCH TRAINING: Decode the FULL sequence [start, mid, end] together
    # Concatenate all three segments
    latents_full = torch.cat([start_lat, latents_mid, end_lat], dim=2)
    
    # Unnormalize the full latent sequence
    latents_mean = torch.tensor(
        vae.config.latents_mean, device=latents_full.device, dtype=latents_full.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=latents_full.device, dtype=latents_full.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_full_unnorm = latents_full / latents_std + latents_mean
    
    # Convert back to VAE dtype before decoding
    latents_full_unnorm = latents_full_unnorm.to(dtype=vae.dtype)

    with torch.no_grad():
        # Decode the entire sequence at once (matches training)
        dec = vae.decode(latents_full_unnorm).sample
        # VAE outputs in [-1, 1] range, convert to [0, 255] for uint8
        dec = dec.clamp(-1, 1)
        dec_np = dec.squeeze(0).cpu().numpy()
        dec_np = np.transpose(dec_np, (1, 2, 3, 0))
        # Convert from [-1, 1] to [0, 255]
        dec_np = ((dec_np / 2.0 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

    # 12) Convert all decoded frames to PIL
    all_frames = [Image.fromarray(frame) for frame in dec_np]
    
    # Calculate frame splits (proportional to original frame counts)
    total_decoded_frames = len(all_frames)
    frac_start = start_duration / float(total_frames)
    frac_mid = mid_frames / float(total_frames)
    
    n_start_decoded = int(round(total_decoded_frames * frac_start))
    n_mid_decoded = int(round(total_decoded_frames * frac_mid))
    n_end_decoded = total_decoded_frames - n_start_decoded - n_mid_decoded
    
    # Ensure we have at least 1 frame in each segment
    n_start_decoded = max(1, n_start_decoded)
    n_mid_decoded = max(1, n_mid_decoded)
    n_end_decoded = max(1, total_decoded_frames - n_start_decoded - n_mid_decoded)
    
    # Extract the segments
    start_decoded = all_frames[:n_start_decoded]
    mid_decoded = all_frames[n_start_decoded:n_start_decoded + n_mid_decoded]
    end_decoded = all_frames[n_start_decoded + n_mid_decoded:]
    
    # 13) Build final output video with all frames
    final_frames = start_decoded + mid_decoded + end_decoded
    export_to_video(final_frames, output_path, fps=out_fps)

    return output_path


def generate_bidirectional_inbetween_from_two_videos(
    base_model_path: str,
    lora_fwd_path: str,
    lora_bwd_path: str,
    fusion_net_path: str,
    start_video_path: str,
    start_frame_index: int,
    start_duration: int,
    end_video_path: str,
    end_frame_index: int,
    end_duration: int,
    mid_frames: int,
    height: int,
    width: int,
    num_inference_steps: int = 50,
    out_fps: int = 24,
    transformer_precision: str = "bf16",
    vae_precision: str = "fp32",
    attn_implementation: str = "sdpa",
    fusion_hidden_dim: int = 256,
    fusion_num_layers: int = 3,
    cnn_feature_dim: int = 64,
    weight_threshold: float = 0.5,
    output_path: str = "bidirectional_inbetween_output.mp4",
    generate_unfused: bool = False,
):
    """
    Generate inbetween frames using bidirectional model with fusion.
    
    This uses two separate LoRAs matching the BidirectionalInbetweenTrainer:
    - Forward LoRA: [start, noisy_mid] with mask [True, False] -> predict mid velocity
    - Backward LoRA: [noisy_mid, end] with mask [False, True] -> predict mid velocity
    - Fusion Network: produces monotonic blending weights via cumulative softmax
    
    The forward and backward predictions are fused using the network weights,
    then the fused velocity is used for the scheduler step.
    
    Args:
        base_model_path: Path to base Wan model
        lora_fwd_path: Path to forward LoRA weights
        lora_bwd_path: Path to backward LoRA weights  
        fusion_net_path: Path to pre-trained fusion network weights
        start_video_path: Path to first video
        start_frame_index: Starting frame index in first video
        start_duration: Number of frames to use from first video
        end_video_path: Path to second video
        end_frame_index: Starting frame index in second video
        end_duration: Number of frames to use from second video
        mid_frames: Number of frames to generate between the two segments
        height: Output height
        width: Output width
        num_inference_steps: Number of denoising steps
        out_fps: Output video FPS
        transformer_precision: Precision for transformer
        vae_precision: Precision for VAE
        attn_implementation: Attention implementation
        fusion_hidden_dim: Hidden dim of fusion network
        fusion_num_layers: Num layers in fusion network
        cnn_feature_dim: CNN feature dim in fusion network
        weight_threshold: Minimum weight to include a direction in fusion
        output_path: Path to save the output video
        generate_unfused: If True, also generate forward-only and backward-only outputs
    
    Returns:
        If generate_unfused is False: Path to the output video
        If generate_unfused is True: Dict with 'fused', 'forward_only', 'backward_only' paths
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load components (single transformer, LoRAs will be swapped during inference)
    vae, transformer, scheduler, fusion_net, lora_fwd_path, lora_bwd_path = load_wan_bidirectional(
        base_model_path=base_model_path,
        lora_fwd_path=lora_fwd_path,
        lora_bwd_path=lora_bwd_path,
        fusion_net_path=fusion_net_path,
        transformer_precision=transformer_precision,
        vae_precision=vae_precision,
        attn_implementation=attn_implementation,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_num_layers=fusion_num_layers,
        cnn_feature_dim=cnn_feature_dim,
    )
    vae.to(device)
    transformer.to(device)
    fusion_net.to(device)
    
    dtype_t = transformer.dtype

    # 2) Load frames from both videos
    start_frames_list = _read_frames_from_video(
        start_video_path, start_frame_index, start_duration
    )
    end_frames_list = _read_frames_from_video(
        end_video_path, end_frame_index, end_duration
    )

    # 3) Convert to tensors and encode
    video_start = _frames_to_tensor(start_frames_list, height, width).to(device)
    video_end = _frames_to_tensor(end_frames_list, height, width).to(device)

    # 4) Encode start and end to normalized latents
    with torch.no_grad():
        enc_start = vae.encode(video_start)
        latents_start = retrieve_latents(enc_start)
        
        enc_end = vae.encode(video_end)
        latents_end = retrieve_latents(enc_end)

        # Normalize using Wan's config
        latents_mean = torch.tensor(
            vae.config.latents_mean, device=latents_start.device, dtype=latents_start.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            vae.config.latents_std, device=latents_start.device, dtype=latents_start.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        
        start_lat = (latents_start - latents_mean) * latents_std
        end_lat = (latents_end - latents_mean) * latents_std
        
        # Convert to transformer dtype
        start_lat = start_lat.to(dtype=dtype_t)
        end_lat = end_lat.to(dtype=dtype_t)

    B, C, T_start_lat, H_lat, W_lat = start_lat.shape
    _, _, T_end_lat, _, _ = end_lat.shape

    # 5) Calculate mid latent frames
    total_frames = start_duration + mid_frames + end_duration
    T_mid_lat = int(round((T_start_lat + T_end_lat) * mid_frames / (start_duration + end_duration)))
    T_mid_lat = max(1, T_mid_lat)  # At least 1 frame

    # 6) Get fusion weights from Fusion Network
    with torch.no_grad():
        w_fwd, w_bwd = fusion_net(start_lat, end_lat, T_mid_lat)
        # w_fwd, w_bwd: [B, T_mid_lat] - weights that sum to 1 per frame

    # 7) Initialize mid latents as noise (same noise for both directions)
    # Use same initial noise for fair comparison across fused/unfused
    initial_noise = torch.randn(
        B, C, T_mid_lat, H_lat, W_lat,
        device=device,
        dtype=dtype_t,
    )
    
    # Create separate latent tracks for forward and backward
    latents_mid_fwd = initial_noise.clone()
    latents_mid_bwd = initial_noise.clone()

    # 8) Scheduler timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # 9) Dummy text encoder hidden states (unconditional)
    text_dim = transformer.config.text_dim
    encoder_hidden_states = torch.zeros(
        B, 1, text_dim, device=device, dtype=dtype_t
    )
    
    # Pre-compute masks (reused each step)
    T_fwd = T_start_lat + T_mid_lat
    T_bwd = T_mid_lat + T_end_lat
    mask_fwd = torch.zeros(B, T_fwd, device=device, dtype=torch.bool)
    mask_fwd[:, :T_start_lat] = True
    mask_bwd = torch.zeros(B, T_bwd, device=device, dtype=torch.bool)
    mask_bwd[:, T_mid_lat:] = True

    # 10) Load LoRAs using PEFT's multi-adapter support
    print("Loading LoRA adapters...")
    
    # Create PeftModel with forward LoRA
    transformer_peft = PeftModel.from_pretrained(transformer, lora_fwd_path, is_trainable=False)
    
    # Load backward LoRA as second adapter
    transformer_peft.load_adapter(lora_bwd_path, adapter_name="backward")
    
    # Move to GPU
    transformer_peft.to(device)
    transformer_peft.eval()
    
    # ============ PHASE 1: Generate forward-only latents ============
    print(f"Phase 1: Forward-only denoising ({len(timesteps)} steps)...")
    transformer_peft.set_adapter("default")  # Forward LoRA
    
    for step_idx, t in enumerate(timesteps):
        if isinstance(t, torch.Tensor):
            timestep = t.unsqueeze(0) if t.dim() == 0 else t.view(-1)
        else:
            timestep = torch.tensor([t], device=device)

        with torch.no_grad():
            model_input_fwd = torch.cat([start_lat, latents_mid_fwd], dim=2)
            
            out_fwd = transformer_peft(
                hidden_states=model_input_fwd,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                conditioning_mask=mask_fwd,
                return_dict=True,
            ).sample
            
            pred_mid_fwd = out_fwd[:, :, T_start_lat:]
            
            step_out = scheduler.step(
                model_output=pred_mid_fwd,
                timestep=t,
                sample=latents_mid_fwd,
            )
            latents_mid_fwd = step_out.prev_sample
    
    # ============ PHASE 2: Generate backward-only latents ============
    print(f"Phase 2: Backward-only denoising ({len(timesteps)} steps)...")
    transformer_peft.set_adapter("backward")  # Backward LoRA
    
    # Reset scheduler for second pass
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    for step_idx, t in enumerate(timesteps):
        if isinstance(t, torch.Tensor):
            timestep = t.unsqueeze(0) if t.dim() == 0 else t.view(-1)
        else:
            timestep = torch.tensor([t], device=device)

        with torch.no_grad():
            model_input_bwd = torch.cat([latents_mid_bwd, end_lat], dim=2)
            
            out_bwd = transformer_peft(
                hidden_states=model_input_bwd,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                conditioning_mask=mask_bwd,
                return_dict=True,
            ).sample
            
            pred_mid_bwd = out_bwd[:, :, :T_mid_lat]
            
            step_out = scheduler.step(
                model_output=pred_mid_bwd,
                timestep=t,
                sample=latents_mid_bwd,
            )
            latents_mid_bwd = step_out.prev_sample
    
    # ============ PHASE 3: Fuse the two generated latents ============
    print("Phase 3: Fusing forward and backward latents...")
    
    # w_fwd, w_bwd: [B, T_mid_lat] -> expand to [B, 1, T_mid_lat, 1, 1]
    w_fwd_exp = w_fwd.view(B, 1, T_mid_lat, 1, 1)
    w_bwd_exp = w_bwd.view(B, 1, T_mid_lat, 1, 1)
    
    # Sparse fusion: only blend when both weights > threshold
    # Otherwise use the dominant direction's prediction
    fwd_above_thresh = (w_fwd > weight_threshold).view(B, 1, T_mid_lat, 1, 1)
    bwd_above_thresh = (w_bwd > weight_threshold).view(B, 1, T_mid_lat, 1, 1)
    
    # Case 1: Both above threshold -> weighted blend
    # Case 2: Only fwd above threshold -> use fwd only
    # Case 3: Only bwd above threshold -> use bwd only  
    # Case 4: Neither above threshold -> use the one with higher weight
    both_above = fwd_above_thresh & bwd_above_thresh
    fwd_only = fwd_above_thresh & ~bwd_above_thresh
    bwd_only = ~fwd_above_thresh & bwd_above_thresh
    
    latents_mid_fused = torch.where(
        both_above,
        w_fwd_exp * latents_mid_fwd + w_bwd_exp * latents_mid_bwd,  # weighted blend
        torch.where(
            fwd_only,
            latents_mid_fwd,  # use forward only
            torch.where(
                bwd_only,
                latents_mid_bwd,  # use backward only
                torch.where(
                    w_fwd_exp > w_bwd_exp,
                    latents_mid_fwd,  # neither above thresh, use dominant
                    latents_mid_bwd
                )
            )
        )
    )

    # 11) Final mid latents

    # 12) Decode separately: [start + mid] and [mid + end]
    # This matches the bidirectional training approach
    
    # Prepare normalization tensors
    latents_mean = torch.tensor(
        vae.config.latents_mean, device=start_lat.device, dtype=start_lat.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=start_lat.device, dtype=start_lat.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    
    # Decode forward path: [start + fused_mid]
    latents_fwd_path = torch.cat([start_lat, latents_mid_fused], dim=2)
    latents_fwd_unnorm = latents_fwd_path / latents_std + latents_mean
    latents_fwd_unnorm = latents_fwd_unnorm.to(dtype=vae.dtype)
    
    with torch.no_grad():
        dec_fwd = vae.decode(latents_fwd_unnorm).sample
        dec_fwd = dec_fwd.clamp(-1, 1)
        dec_fwd_np = dec_fwd.squeeze(0).cpu().numpy()
        dec_fwd_np = np.transpose(dec_fwd_np, (1, 2, 3, 0))
        dec_fwd_np = ((dec_fwd_np / 2.0 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    
    # Decode backward path: [fused_mid + end]
    latents_bwd_path = torch.cat([latents_mid_fused, end_lat], dim=2)
    latents_bwd_unnorm = latents_bwd_path / latents_std + latents_mean
    latents_bwd_unnorm = latents_bwd_unnorm.to(dtype=vae.dtype)
    
    with torch.no_grad():
        dec_bwd = vae.decode(latents_bwd_unnorm).sample
        dec_bwd = dec_bwd.clamp(-1, 1)
        dec_bwd_np = dec_bwd.squeeze(0).cpu().numpy()
        dec_bwd_np = np.transpose(dec_bwd_np, (1, 2, 3, 0))
        dec_bwd_np = ((dec_bwd_np / 2.0 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    
    # 13) Split decoded frames and concatenate
    # Forward path: [start + mid] -> extract start frames and first half of mid
    # Backward path: [mid + end] -> extract second half of mid and end frames
    
    # Calculate frame splits for forward path
    total_fwd_frames = len(dec_fwd_np)
    frac_start_fwd = start_duration / float(start_duration + mid_frames)
    n_start_fwd = int(round(total_fwd_frames * frac_start_fwd))
    n_start_fwd = max(1, n_start_fwd)
    n_mid_fwd = total_fwd_frames - n_start_fwd
    
    # Calculate frame splits for backward path
    total_bwd_frames = len(dec_bwd_np)
    frac_mid_bwd = mid_frames / float(mid_frames + end_duration)
    n_mid_bwd = int(round(total_bwd_frames * frac_mid_bwd))
    n_mid_bwd = max(1, n_mid_bwd)
    n_end_bwd = total_bwd_frames - n_mid_bwd
    
    # Extract frames from forward path: all start + first half of mid
    start_frames_decoded = [Image.fromarray(frame) for frame in dec_fwd_np[:n_start_fwd]]
    mid_fwd_frames = [Image.fromarray(frame) for frame in dec_fwd_np[n_start_fwd:]]
    
    # Extract frames from backward path: second half of mid + all end
    mid_bwd_frames = [Image.fromarray(frame) for frame in dec_bwd_np[:n_mid_bwd]]
    end_frames_decoded = [Image.fromarray(frame) for frame in dec_bwd_np[n_mid_bwd:]]
    
    # Split mid frames: first half from fwd, second half from bwd
    mid_fwd_half = len(mid_fwd_frames) // 2
    mid_bwd_half = len(mid_bwd_frames) - len(mid_bwd_frames) // 2
    
    mid_frames_decoded = mid_fwd_frames[:mid_fwd_half] + mid_bwd_frames[-mid_bwd_half:]
    
    # 14) Build final output video
    final_frames = start_frames_decoded + mid_frames_decoded + end_frames_decoded
    export_to_video(final_frames, output_path, fps=out_fps)
    
    # 15) If generating unfused outputs, decode and save them too
    if generate_unfused:
        output_dir = os.path.dirname(output_path)
        output_base = os.path.splitext(os.path.basename(output_path))[0]
        
        # --- Decode forward-only latents ---
        latents_fwd_only_path = torch.cat([start_lat, latents_mid_fwd], dim=2)
        latents_fwd_only_unnorm = latents_fwd_only_path / latents_std + latents_mean
        latents_fwd_only_unnorm = latents_fwd_only_unnorm.to(dtype=vae.dtype)
        
        with torch.no_grad():
            dec_fwd_only = vae.decode(latents_fwd_only_unnorm).sample
            dec_fwd_only = dec_fwd_only.clamp(-1, 1)
            dec_fwd_only_np = dec_fwd_only.squeeze(0).cpu().numpy()
            dec_fwd_only_np = np.transpose(dec_fwd_only_np, (1, 2, 3, 0))
            dec_fwd_only_np = ((dec_fwd_only_np / 2.0 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        
        # Also decode [fwd_only_mid + end] for complete video
        latents_fwd_only_bwd_path = torch.cat([latents_mid_fwd, end_lat], dim=2)
        latents_fwd_only_bwd_unnorm = latents_fwd_only_bwd_path / latents_std + latents_mean
        latents_fwd_only_bwd_unnorm = latents_fwd_only_bwd_unnorm.to(dtype=vae.dtype)
        
        with torch.no_grad():
            dec_fwd_only_bwd = vae.decode(latents_fwd_only_bwd_unnorm).sample
            dec_fwd_only_bwd = dec_fwd_only_bwd.clamp(-1, 1)
            dec_fwd_only_bwd_np = dec_fwd_only_bwd.squeeze(0).cpu().numpy()
            dec_fwd_only_bwd_np = np.transpose(dec_fwd_only_bwd_np, (1, 2, 3, 0))
            dec_fwd_only_bwd_np = ((dec_fwd_only_bwd_np / 2.0 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        
        # Build forward-only video: start + mid + end
        start_fwd_only = [Image.fromarray(f) for f in dec_fwd_only_np[:n_start_fwd]]
        mid_fwd_only = [Image.fromarray(f) for f in dec_fwd_only_np[n_start_fwd:]]
        end_fwd_only = [Image.fromarray(f) for f in dec_fwd_only_bwd_np[n_mid_bwd:]]
        fwd_only_frames = start_fwd_only + mid_fwd_only + end_fwd_only
        
        fwd_only_output_path = os.path.join(output_dir, f"{output_base}_forward_only.mp4")
        export_to_video(fwd_only_frames, fwd_only_output_path, fps=out_fps)
        
        # --- Decode backward-only latents ---
        latents_bwd_only_fwd_path = torch.cat([start_lat, latents_mid_bwd], dim=2)
        latents_bwd_only_fwd_unnorm = latents_bwd_only_fwd_path / latents_std + latents_mean
        latents_bwd_only_fwd_unnorm = latents_bwd_only_fwd_unnorm.to(dtype=vae.dtype)
        
        with torch.no_grad():
            dec_bwd_only_fwd = vae.decode(latents_bwd_only_fwd_unnorm).sample
            dec_bwd_only_fwd = dec_bwd_only_fwd.clamp(-1, 1)
            dec_bwd_only_fwd_np = dec_bwd_only_fwd.squeeze(0).cpu().numpy()
            dec_bwd_only_fwd_np = np.transpose(dec_bwd_only_fwd_np, (1, 2, 3, 0))
            dec_bwd_only_fwd_np = ((dec_bwd_only_fwd_np / 2.0 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        
        latents_bwd_only_bwd_path = torch.cat([latents_mid_bwd, end_lat], dim=2)
        latents_bwd_only_unnorm = latents_bwd_only_bwd_path / latents_std + latents_mean
        latents_bwd_only_unnorm = latents_bwd_only_unnorm.to(dtype=vae.dtype)
        
        with torch.no_grad():
            dec_bwd_only = vae.decode(latents_bwd_only_unnorm).sample
            dec_bwd_only = dec_bwd_only.clamp(-1, 1)
            dec_bwd_only_np = dec_bwd_only.squeeze(0).cpu().numpy()
            dec_bwd_only_np = np.transpose(dec_bwd_only_np, (1, 2, 3, 0))
            dec_bwd_only_np = ((dec_bwd_only_np / 2.0 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        
        # Build backward-only video: start + mid + end
        start_bwd_only = [Image.fromarray(f) for f in dec_bwd_only_fwd_np[:n_start_fwd]]
        mid_bwd_only = [Image.fromarray(f) for f in dec_bwd_only_np[:n_mid_bwd]]
        end_bwd_only = [Image.fromarray(f) for f in dec_bwd_only_np[n_mid_bwd:]]
        bwd_only_frames = start_bwd_only + mid_bwd_only + end_bwd_only
        
        bwd_only_output_path = os.path.join(output_dir, f"{output_base}_backward_only.mp4")
        export_to_video(bwd_only_frames, bwd_only_output_path, fps=out_fps)
        
        print(f"Saved unfused outputs:")
        print(f"  Forward-only: {fwd_only_output_path}")
        print(f"  Backward-only: {bwd_only_output_path}")
        
        return {
            'fused': output_path,
            'forward_only': fwd_only_output_path,
            'backward_only': bwd_only_output_path,
        }

    return output_path