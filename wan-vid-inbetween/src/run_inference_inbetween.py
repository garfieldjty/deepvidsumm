# src/run_inference_inbetween.py
"""
Inference script for video inbetweening.

Supports both:
- Standard (unidirectional) model: uses --lora_path only
- Bidirectional model: uses --lora_path + --fusion_mlp_path

Examples:
    # Standard model
    python -m src.run_inference_inbetween --lora_path ./outputs/inbetween_lora \\
        --start_video_path video.mp4 --start_frame_index 0 --start_duration 30 \\
        --end_video_path video.mp4 --end_frame_index 90 --end_duration 30
    
    # Bidirectional model  
    python -m src.run_inference_inbetween --lora_path ./outputs/bidirectional_lora/lora \\
        --fusion_mlp_path ./outputs/bidirectional_lora/fusion_mlp.pt \\
        --start_video_path video.mp4 --start_frame_index 0 --start_duration 30 \\
        --end_video_path video.mp4 --end_frame_index 90 --end_duration 30
"""
import argparse
from accelerate import Accelerator

from .utils import load_config
from .inference import (
    generate_inbetween_from_two_videos,
    generate_bidirectional_inbetween_from_two_videos,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Wan2.2 inbetweening inference (supports both unidirectional and bidirectional models)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_inbetween_config.yaml",
        help="Path to YAML config file",
    )
    
    # Required parameters (can override config)
    parser.add_argument("--lora_path", type=str, help="Path to trained LoRA weights")
    parser.add_argument("--start_video_path", type=str, help="Path to first/start video")
    parser.add_argument("--start_frame_index", type=int, help="Starting frame index in start video")
    parser.add_argument("--start_duration", type=int, help="Number of frames to use from start video")
    parser.add_argument("--end_video_path", type=str, help="Path to second/end video")
    parser.add_argument("--end_frame_index", type=int, help="Starting frame index in end video")
    parser.add_argument("--end_duration", type=int, help="Number of frames to use from end video")
    
    # Optional parameters (override config defaults)
    parser.add_argument("--base_model_path", type=str, help="Base model path")
    parser.add_argument("--mid_frames", type=int, help="Number of frames to generate")
    parser.add_argument("--height", type=int, help="Output height")
    parser.add_argument("--width", type=int, help="Output width")
    parser.add_argument("--num_inference_steps", type=int, help="Number of inference steps")
    parser.add_argument("--out_fps", type=int, help="Output FPS")
    parser.add_argument("--output_path", type=str, help="Output video path")
    parser.add_argument("--transformer_precision", type=str, choices=["bf16", "fp16", "no"], help="Transformer precision")
    parser.add_argument("--vae_precision", type=str, choices=["fp32", "fp16"], help="VAE precision")
    
    # Bidirectional model parameters
    parser.add_argument(
        "--fusion_mlp_path", 
        type=str, 
        default=None,
        help="Path to fusion MLP weights (enables bidirectional mode)"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation to use",
    )
    parser.add_argument(
        "--fusion_hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of fusion MLP (if bidirectional)",
    )
    parser.add_argument(
        "--fusion_num_layers",
        type=int,
        default=3,
        help="Number of layers in fusion MLP (if bidirectional)",
    )
    parser.add_argument(
        "--cnn_feature_dim",
        type=int,
        default=64,
        help="CNN feature dimension in fusion MLP (if bidirectional)",
    )

    args = parser.parse_args()

    # Initialize accelerator for distributed inference support
    accelerator = Accelerator()
    
    # Check if using bidirectional mode
    use_bidirectional = args.fusion_mlp_path is not None
    
    # Load config file
    cfg = load_config(args.config)
    
    # Get inference defaults from config
    inference_cfg = cfg.get("inference", {})
    
    # Required parameters: prioritize CLI args, then fail if missing
    if args.lora_path is None:
        raise ValueError("--lora_path is required")
    if args.start_video_path is None:
        raise ValueError("--start_video_path is required")
    if args.start_frame_index is None:
        raise ValueError("--start_frame_index is required")
    if args.start_duration is None:
        raise ValueError("--start_duration is required")
    if args.end_video_path is None:
        raise ValueError("--end_video_path is required")
    if args.end_frame_index is None:
        raise ValueError("--end_frame_index is required")
    if args.end_duration is None:
        raise ValueError("--end_duration is required")
    
    # Merge config with CLI args (CLI args take precedence)
    base_model_path = args.base_model_path or cfg.get("base_model_path")
    mid_frames = args.mid_frames if args.mid_frames is not None else inference_cfg.get("mid_frames", 60)
    height = args.height if args.height is not None else inference_cfg.get("height", 480)
    width = args.width if args.width is not None else inference_cfg.get("width", 832)
    num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else inference_cfg.get("num_inference_steps", 50)
    out_fps = args.out_fps if args.out_fps is not None else inference_cfg.get("out_fps", 24)
    output_path = args.output_path or inference_cfg.get("output_path", "inbetween_output.mp4")
    transformer_precision = args.transformer_precision or cfg.get("transformer_precision", "bf16")
    vae_precision = args.vae_precision or cfg.get("vae_precision", "fp32")
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print("VIDEO INBETWEENING INFERENCE")
        print(f"{'='*60}")
        print(f"Mode: {'BIDIRECTIONAL' if use_bidirectional else 'STANDARD (unidirectional)'}")
        print(f"Config: {args.config}")
        print(f"Base model: {base_model_path}")
        print(f"LoRA path: {args.lora_path}")
        if use_bidirectional:
            print(f"Fusion MLP: {args.fusion_mlp_path}")
        print(f"Start video: {args.start_video_path} (frame {args.start_frame_index}, duration {args.start_duration})")
        print(f"End video: {args.end_video_path} (frame {args.end_frame_index}, duration {args.end_duration})")
        print(f"Frames to generate: {mid_frames}")
        print(f"Resolution: {height}x{width}")
        print(f"Steps: {num_inference_steps}, FPS: {out_fps}")
        print(f"{'='*60}\n")

    if use_bidirectional:
        # Bidirectional model inference
        out = generate_bidirectional_inbetween_from_two_videos(
            base_model_path=base_model_path,
            lora_path=args.lora_path,
            fusion_mlp_path=args.fusion_mlp_path,
            start_video_path=args.start_video_path,
            start_frame_index=args.start_frame_index,
            start_duration=args.start_duration,
            end_video_path=args.end_video_path,
            end_frame_index=args.end_frame_index,
            end_duration=args.end_duration,
            mid_frames=mid_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            out_fps=out_fps,
            transformer_precision=transformer_precision,
            vae_precision=vae_precision,
            attn_implementation=args.attn_implementation,
            fusion_hidden_dim=args.fusion_hidden_dim,
            fusion_num_layers=args.fusion_num_layers,
            cnn_feature_dim=args.cnn_feature_dim,
            output_path=output_path,
        )
    else:
        # Standard unidirectional model inference
        out = generate_inbetween_from_two_videos(
            base_model_path=base_model_path,
            lora_path=args.lora_path,
            start_video_path=args.start_video_path,
            start_frame_index=args.start_frame_index,
            start_duration=args.start_duration,
            end_video_path=args.end_video_path,
            end_frame_index=args.end_frame_index,
            end_duration=args.end_duration,
            mid_frames=mid_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            out_fps=out_fps,
            transformer_precision=transformer_precision,
            vae_precision=vae_precision,
            output_path=output_path,
        )
    
    if accelerator.is_main_process:
        print(f"\n✓ Saved: {out}")


if __name__ == "__main__":
    main()
