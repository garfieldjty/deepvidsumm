"""
Example inference script for LongCat Video In-Between task.

This script demonstrates how to use a trained LongCat model to generate 
in-between frames given starting and ending clips.

Usage:
    uv run python inference_inbetween.py \
        --model_path ./models/train/LongCat-Video-InBetween_lora \
        --start_frames path/to/start_frame1.png path/to/start_frame2.png \
        --end_frames path/to/end_frame1.png path/to/end_frame2.png \
        --prompt "A video description" \
        --output_path ./output/generated_video.mp4 \
        --num_frames 49
"""

import torch
import argparse
from PIL import Image
from diffsynth import ModelManager, save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, model_fn_longcat_video_inbetween


def load_images(image_paths):
    """Load images from paths."""
    return [Image.open(path).convert("RGB") for path in image_paths]


def main():
    parser = argparse.ArgumentParser(description="LongCat Video In-Between Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained LoRA model")
    parser.add_argument("--base_model_id", type=str, default="meituan-longcat/LongCat-Video", help="Base LongCat model ID")
    parser.add_argument("--start_frames", nargs="+", required=True, help="Paths to starting frames")
    parser.add_argument("--end_frames", nargs="+", required=True, help="Paths to ending frames")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the video")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--output_path", type=str, default="./output_inbetween.mp4", help="Output video path")
    parser.add_argument("--num_frames", type=int, default=49, help="Total number of frames to generate")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="Classifier-free guidance scale")
    parser.add_argument("--sigma_shift", type=float, default=1.0, help="Sigma shift for scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tiled", action="store_true", default=True, help="Use tiled VAE encoding/decoding")
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality (1-10)")
    
    args = parser.parse_args()
    
    # Load the pipeline with base models
    print("Loading base models...")
    from diffsynth.pipelines.wan_video_new import ModelConfig
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="meituan-longcat/LongCat-Video", origin_file_pattern="dit/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ]
    )
    pipe.enable_vram_management()
    
    # Load the trained LoRA weights
    print(f"Loading LoRA from {args.model_path}...")
    import os
    # If model_path is a directory, find the latest checkpoint
    if os.path.isdir(args.model_path):
        checkpoints = sorted([f for f in os.listdir(args.model_path) if f.endswith('.safetensors')])
        if checkpoints:
            lora_path = os.path.join(args.model_path, checkpoints[-1])
            print(f"Using checkpoint: {checkpoints[-1]}")
        else:
            raise ValueError(f"No .safetensors files found in {args.model_path}")
    else:
        lora_path = args.model_path
    
    pipe.load_lora(
        pipe.dit,
        lora_config=lora_path,
        alpha=1.0,
    )
    
    # Override model function to use in-between mode
    pipe.model_fn = model_fn_longcat_video_inbetween
    
    # Load start and end frames
    print("Loading input frames...")
    start_frames = load_images(args.start_frames)
    end_frames = load_images(args.end_frames)
    
    print(f"Generating video with {len(start_frames)} start frames and {len(end_frames)} end frames...")
    print(f"Total frames to generate: {args.num_frames}")
    print(f"In-between frames: {args.num_frames - len(start_frames) - len(end_frames)}")
    
    # Generate the video
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        longcat_start_video=start_frames,
        longcat_end_video=end_frames,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        tiled=args.tiled,
    )
    
    # Save the output
    print(f"Saving video to {args.output_path}...")
    save_video(video, args.output_path, fps=args.fps, quality=args.quality)
    print("Done!")


if __name__ == "__main__":
    main()
