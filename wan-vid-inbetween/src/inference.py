# src/inference.py

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
from .trainer import CumulativeSoftmaxFusionMLP


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
    lora_path: str,
    fusion_mlp_path: str,
    transformer_precision: str = "bf16",
    vae_precision: str = "fp32",
    attn_implementation: str = "sdpa",
    fusion_hidden_dim: int = 256,
    fusion_num_layers: int = 3,
    cnn_feature_dim: int = 64,
):
    """
    Load Wan model with bidirectional LoRA and fusion MLP.
    
    Args:
        base_model_path: Path to base Wan model
        lora_path: Path to trained LoRA weights (single LoRA for both directions)
        fusion_mlp_path: Path to pre-trained fusion MLP weights
        transformer_precision: Precision for transformer
        vae_precision: Precision for VAE
        attn_implementation: Attention implementation
        fusion_hidden_dim: Hidden dim of fusion MLP (must match training)
        fusion_num_layers: Num layers in fusion MLP (must match training)
        cnn_feature_dim: CNN feature dim in fusion MLP (must match training)
    
    Returns:
        vae, transformer, scheduler, fusion_mlp
    """
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
    
    # Load fusion MLP
    latent_dim = vae.config.z_dim
    fusion_mlp = CumulativeSoftmaxFusionMLP(
        latent_dim=latent_dim,
        hidden_dim=fusion_hidden_dim,
        num_layers=fusion_num_layers,
        cnn_feature_dim=cnn_feature_dim,
    )
    fusion_mlp.load_state_dict(torch.load(fusion_mlp_path, map_location="cpu"))
    fusion_mlp.eval()
    fusion_mlp.to(dtype_t)

    return vae, transformer, scheduler, fusion_mlp


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
    lora_path: str,
    fusion_mlp_path: str,
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
    weight_threshold: float = 0.4,
    output_path: str = "bidirectional_inbetween_output.mp4",
):
    """
    Generate inbetween frames using bidirectional model with fusion.
    
    This uses:
    - Single LoRA for both forward and backward denoising passes
    - A pre-trained fusion MLP that produces monotonic blending weights
    - Forward pass: conditions on start frames
    - Backward pass: conditions on end frames
    - Fusion: weighted combination using cumulative softmax weights
    - Sparse fusion: only blend when both weights > threshold
    
    Args:
        base_model_path: Path to base Wan model
        lora_path: Path to trained bidirectional LoRA weights
        fusion_mlp_path: Path to pre-trained fusion MLP weights
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
        fusion_hidden_dim: Hidden dim of fusion MLP
        fusion_num_layers: Num layers in fusion MLP
        cnn_feature_dim: CNN feature dim in fusion MLP
        weight_threshold: Minimum weight to include a direction in fusion (default 0.3)
        output_path: Path to save the output video
    
    Returns:
        Path to the output video
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load components + LoRA + Fusion MLP
    vae, transformer, scheduler, fusion_mlp = load_wan_bidirectional(
        base_model_path=base_model_path,
        lora_path=lora_path,
        fusion_mlp_path=fusion_mlp_path,
        transformer_precision=transformer_precision,
        vae_precision=vae_precision,
        attn_implementation=attn_implementation,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_num_layers=fusion_num_layers,
        cnn_feature_dim=cnn_feature_dim,
    )
    vae.to(device)
    transformer.to(device)
    fusion_mlp.to(device)

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

        # Normalize
        latents_mean = torch.tensor(
            vae.config.latents_mean, device=latents_start.device, dtype=latents_start.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            vae.config.latents_std, device=latents_start.device, dtype=latents_start.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        
        start_lat = (latents_start - latents_mean) * latents_std
        end_lat = (latents_end - latents_mean) * latents_std
        
        # Convert to transformer dtype
        start_lat = start_lat.to(dtype=transformer.dtype)
        end_lat = end_lat.to(dtype=transformer.dtype)

    B, C, T_start_lat, H_lat, W_lat = start_lat.shape
    _, _, T_end_lat, _, _ = end_lat.shape

    # 5) Calculate mid latent frames
    total_frames = start_duration + mid_frames + end_duration
    T_mid_lat = int(round((T_start_lat + T_end_lat) * mid_frames / (start_duration + end_duration)))
    T_mid_lat = max(1, T_mid_lat)  # At least 1 frame

    # 6) Get fusion weights from MLP
    with torch.no_grad():
        w_fwd, w_bwd = fusion_mlp(start_lat, end_lat, T_mid_lat)
        # w_fwd, w_bwd: [B, T_mid_lat] - weights that sum to 1 per frame

    # 7) Initialize mid latents as noise (same noise for both directions)
    latents_mid_fwd = torch.randn(
        B, C, T_mid_lat, H_lat, W_lat,
        device=device,
        dtype=transformer.dtype,
    )
    latents_mid_bwd = latents_mid_fwd.clone()  # Start from same noise

    # 8) Scheduler timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # 9) Dummy text encoder hidden states
    text_dim = transformer.config.text_dim
    encoder_hidden_states = torch.zeros(
        B, 1, text_dim, device=device, dtype=transformer.dtype
    )

    # 10) Denoising loop - run forward and backward passes, fuse predictions
    for t in timesteps:
        # Ensure timestep is properly shaped as 1D tensor
        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                timestep = t.unsqueeze(0)
            else:
                timestep = t.view(-1)
        else:
            timestep = torch.tensor([t], device=device)

        with torch.no_grad():
            # ---- Forward pass: [start, mid] ----
            # Conditions on start, generates continuation
            model_input_fwd = torch.cat([start_lat, latents_mid_fwd], dim=2)
            T_fwd = T_start_lat + T_mid_lat
            
            conditioning_mask_fwd = torch.zeros(B, T_fwd, device=device, dtype=torch.bool)
            conditioning_mask_fwd[:, :T_start_lat] = True  # Start is conditioning

            out_fwd = transformer(
                hidden_states=model_input_fwd,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                conditioning_mask=conditioning_mask_fwd,
                return_dict=True,
            ).sample
            
            # Extract mid prediction from forward pass
            pred_mid_fwd = out_fwd[:, :, T_start_lat:]  # [B, C, T_mid_lat, H, W]

            # ---- Backward pass: [mid, end] ----
            # Conditions on end, generates what comes before
            model_input_bwd = torch.cat([latents_mid_bwd, end_lat], dim=2)
            T_bwd = T_mid_lat + T_end_lat
            
            conditioning_mask_bwd = torch.zeros(B, T_bwd, device=device, dtype=torch.bool)
            conditioning_mask_bwd[:, T_mid_lat:] = True  # End is conditioning

            out_bwd = transformer(
                hidden_states=model_input_bwd,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                conditioning_mask=conditioning_mask_bwd,
                return_dict=True,
            ).sample
            
            # Extract mid prediction from backward pass
            pred_mid_bwd = out_bwd[:, :, :T_mid_lat]  # [B, C, T_mid_lat, H, W]

            # ---- Fuse predictions using weights ----
            # w_fwd, w_bwd: [B, T_mid_lat] -> expand to [B, 1, T_mid_lat, 1, 1]
            w_fwd_exp = w_fwd.view(B, 1, T_mid_lat, 1, 1)
            w_bwd_exp = w_bwd.view(B, 1, T_mid_lat, 1, 1)
            
            # Sparse fusion: only blend when both weights > threshold
            # Otherwise use the dominant direction's prediction
            fwd_above_thresh = (w_fwd > weight_threshold).view(B, 1, T_mid_lat, 1, 1)  # [B, 1, T, 1, 1]
            bwd_above_thresh = (w_bwd > weight_threshold).view(B, 1, T_mid_lat, 1, 1)
            
            # Case 1: Both above threshold -> fuse with weights
            # Case 2: Only fwd above threshold -> use fwd only
            # Case 3: Only bwd above threshold -> use bwd only
            # Case 4: Neither above threshold (rare) -> use the one with higher weight
            both_above = fwd_above_thresh & bwd_above_thresh
            fwd_only = fwd_above_thresh & ~bwd_above_thresh
            bwd_only = ~fwd_above_thresh & bwd_above_thresh
            neither = ~fwd_above_thresh & ~bwd_above_thresh
            
            # Build fused prediction based on conditions
            pred_mid_fused = torch.where(
                both_above,
                w_fwd_exp * pred_mid_fwd + w_bwd_exp * pred_mid_bwd,  # weighted blend
                torch.where(
                    fwd_only,
                    pred_mid_fwd,  # use forward only
                    torch.where(
                        bwd_only,
                        pred_mid_bwd,  # use backward only
                        torch.where(
                            w_fwd_exp > w_bwd_exp,
                            pred_mid_fwd,  # neither above thresh, use dominant
                            pred_mid_bwd
                        )
                    )
                )
            )

            # ---- Scheduler step for mid latents ----
            # Average the current mid latents for the step
            latents_mid_avg = 0.5 * (latents_mid_fwd + latents_mid_bwd)
            
            step_out = scheduler.step(
                model_output=pred_mid_fused,
                timestep=t,
                sample=latents_mid_avg,
            )
            
            # Update both mid latent tracks to the same value
            latents_mid_fwd = step_out.prev_sample
            latents_mid_bwd = step_out.prev_sample.clone()

    # 11) Final mid latents
    latents_mid = latents_mid_fwd  # Both are the same at this point

    # 12) Decode the FULL sequence [start, mid, end] together
    latents_full = torch.cat([start_lat, latents_mid, end_lat], dim=2)
    
    # Unnormalize
    latents_mean = torch.tensor(
        vae.config.latents_mean, device=latents_full.device, dtype=latents_full.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=latents_full.device, dtype=latents_full.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_full_unnorm = latents_full / latents_std + latents_mean
    
    latents_full_unnorm = latents_full_unnorm.to(dtype=vae.dtype)

    with torch.no_grad():
        dec = vae.decode(latents_full_unnorm).sample
        dec = dec.clamp(-1, 1)
        dec_np = dec.squeeze(0).cpu().numpy()
        dec_np = np.transpose(dec_np, (1, 2, 3, 0))
        dec_np = ((dec_np / 2.0 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

    # 13) Convert to PIL and split into segments
    all_frames = [Image.fromarray(frame) for frame in dec_np]
    
    total_decoded_frames = len(all_frames)
    frac_start = start_duration / float(total_frames)
    frac_mid = mid_frames / float(total_frames)
    
    n_start_decoded = int(round(total_decoded_frames * frac_start))
    n_mid_decoded = int(round(total_decoded_frames * frac_mid))
    n_end_decoded = total_decoded_frames - n_start_decoded - n_mid_decoded
    
    n_start_decoded = max(1, n_start_decoded)
    n_mid_decoded = max(1, n_mid_decoded)
    n_end_decoded = max(1, total_decoded_frames - n_start_decoded - n_mid_decoded)
    
    start_decoded = all_frames[:n_start_decoded]
    mid_decoded = all_frames[n_start_decoded:n_start_decoded + n_mid_decoded]
    end_decoded = all_frames[n_start_decoded + n_mid_decoded:]
    
    # 14) Build final output video
    final_frames = start_decoded + mid_decoded + end_decoded
    export_to_video(final_frames, output_path, fps=out_fps)

    return output_path