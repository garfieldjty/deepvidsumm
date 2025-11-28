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
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load the pipeline
    print("Loading models...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda"
    )
    
    # Load the trained LoRA weights
    print(f"Loading LoRA from {args.model_path}...")
    pipe.load_lora(
        pipe.dit,
        lora_config=args.model_path,
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
        seed=args.seed,
    )
    
    # Save the output
    print(f"Saving video to {args.output_path}...")
    save_video(video, args.output_path, fps=24)
    print("Done!")


if __name__ == "__main__":
    main()
