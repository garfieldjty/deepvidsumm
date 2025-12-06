"""
Inference script for Wan2.2-TI2V-5B video inbetweening.

This script runs inference with a trained Wan2.2-TI2V-5B model for 
generating intermediate video frames given start and end frames.

Usage:
    python inference_inbetween_ti2v.py \
        --input_video /path/to/input.mp4 \
        --output_path /path/to/output.mp4 \
        --lora_path ./models/train/Wan2.2-TI2V-InBetween_lora/epoch-X.safetensors \
        --num_start_frames 8 \
        --num_end_frames 8
"""

import torch
import argparse
from PIL import Image
from pathlib import Path

from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


def model_fn_wan_ti2v_inbetween(
    dit,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    image_latents: torch.Tensor = None,
    start_latents: torch.Tensor = None,
    end_latents: torch.Tensor = None,
    num_start_frames: int = 1,
    num_end_frames: int = 1,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs
):
    """
    Model function for Wan2.2-TI2V-5B video in-between task.
    """
    # Create conditioning mask for the latents
    B, C, T, H, W = latents.shape
    
    # Replace start and end portions of latents with clean conditioned latents
    if start_latents is not None:
        t_start = start_latents.shape[2]
        latents = latents.clone()
        latents[:, :, :t_start] = start_latents
    
    if end_latents is not None:
        t_end = end_latents.shape[2]
        latents = latents.clone()
        latents[:, :, -t_end:] = end_latents
    
    # Prepare context embeddings
    context = context.unsqueeze(0) if context.dim() == 2 else context
    encoder_attention_mask = torch.any(context != 0, dim=-1)[:, 0].to(torch.int64)
    
    # Call the dit model
    output = dit(
        latents,
        timestep,
        context,
        encoder_attention_mask,
        image_latents=image_latents,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    
    output = -output
    output = output.to(latents.dtype)
    return output


def load_video_frames(video_path: str, num_frames: int, height: int, width: int):
    """Load video frames and resize them."""
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        frame = cv2.resize(frame, (width, height))
        frames.append(Image.fromarray(frame))
    
    cap.release()
    
    # Pad if necessary
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    return frames


def run_inbetween_inference(
    pipe,
    start_frames: list,
    end_frames: list,
    num_frames: int,
    height: int,
    width: int,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = 42,
    num_inference_steps: int = 30,
    cfg_scale: float = 5.0,
):
    """Run video inbetweening inference."""
    import numpy as np
    
    # Override model function
    pipe.model_fn = model_fn_wan_ti2v_inbetween
    
    # Encode start and end frames to latents
    def encode_frames(frames):
        if isinstance(frames, list):
            frames_np = np.stack([np.array(f) for f in frames], axis=0)
        else:
            frames_np = frames
        
        # Normalize to [-1, 1]
        frames_np = frames_np.astype(np.float32) / 127.5 - 1.0
        
        # [T, H, W, C] -> [1, C, T, H, W]
        frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0)
        frames_tensor = frames_tensor.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        with torch.no_grad():
            latents = pipe.vae.encode(frames_tensor)
        
        return latents
    
    start_latents = encode_frames(start_frames)
    end_latents = encode_frames(end_frames)
    
    # Run inference
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=start_frames[0],  # First frame as TI2V input
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        tiled=True,
        # Pass inbetween conditioning through extra parameters
        start_latents=start_latents,
        end_latents=end_latents,
        num_start_frames=len(start_frames),
        num_end_frames=len(end_frames),
    )
    
    return video


def main():
    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B Video Inbetweening Inference")
    
    # Input/Output
    parser.add_argument("--input_video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_path", type=str, default="./output_inbetween.mp4", help="Output video path")
    
    # Model
    parser.add_argument("--lora_path", type=str, default=None, help="Path to trained LoRA weights")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha scaling")
    
    # Resolution
    parser.add_argument("--height", type=int, default=480, help="Output height")
    parser.add_argument("--width", type=int, default=832, help="Output width")
    
    # Frame configuration
    parser.add_argument("--num_frames", type=int, default=49, help="Total number of output frames")
    parser.add_argument("--num_start_frames", type=int, default=8, help="Number of start conditioning frames")
    parser.add_argument("--num_end_frames", type=int, default=8, help="Number of end conditioning frames")
    parser.add_argument("--start_frame_idx", type=int, default=0, help="Starting frame index in input video")
    parser.add_argument("--end_frame_idx", type=int, default=None, help="Ending frame index (default: start + num_frames)")
    
    # Prompt
    parser.add_argument("--prompt", type=str, default="", help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    
    # Inference
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS")
    
    args = parser.parse_args()
    
    print("=== Loading Wan2.2-TI2V-5B Pipeline ===")
    
    # Load pipeline
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu"
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu"
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="Wan2.2_VAE.pth",
                offload_device="cpu"
            ),
        ],
    )
    
    # Load LoRA if provided
    if args.lora_path:
        print(f"Loading LoRA weights from: {args.lora_path}")
        state_dict = load_state_dict(args.lora_path)
        pipe.load_lora(pipe.dit, state_dict=state_dict, alpha=args.lora_alpha)
    
    pipe.enable_vram_management()
    
    print("=== Loading Video Frames ===")
    
    # Calculate frame indices
    end_frame_idx = args.end_frame_idx if args.end_frame_idx else args.start_frame_idx + args.num_frames
    total_frames_needed = end_frame_idx - args.start_frame_idx
    
    # Load all frames from video
    video_data = VideoData(
        args.input_video,
        height=args.height,
        width=args.width,
        start_frame=args.start_frame_idx,
        num_frames=total_frames_needed
    )
    
    # Extract start and end frames
    start_frames = [video_data[i] for i in range(args.num_start_frames)]
    end_frames = [video_data[-(args.num_end_frames - i)] for i in range(args.num_end_frames)]
    
    print(f"Start frames: {len(start_frames)}, End frames: {len(end_frames)}")
    print(f"Generating {args.num_frames} total frames at {args.height}x{args.width}")
    
    print("=== Running Inference ===")
    
    # Run inbetweening
    video = run_inbetween_inference(
        pipe=pipe,
        start_frames=start_frames,
        end_frames=end_frames,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
    )
    
    print(f"=== Saving Output to {args.output_path} ===")
    save_video(video, args.output_path, fps=args.fps, quality=5)
    
    print("Done!")


if __name__ == "__main__":
    main()
