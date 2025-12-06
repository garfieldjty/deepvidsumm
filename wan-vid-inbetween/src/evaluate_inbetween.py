#!/usr/bin/env python3
"""
Evaluation script for inbetweening model.

This script:
1. Reads cut annotations from a JSON file
2. For each cut, extracts video segments before and after the cut
3. Generates inbetween frames using the trained model
4. Compares generated frames with ground truth middle frames
5. Computes quality metrics (SSIM, VMAF, PSNR)
6. Saves results and video clips to an output folder

Supports both:
- Standard (unidirectional) model: --lora_path only
- Bidirectional model: --lora_path + --fusion_mlp_path
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from accelerate import Accelerator

from .inference import (
    generate_inbetween_from_two_videos,
    generate_bidirectional_inbetween_from_two_videos,
)
from .utils import load_config


def extract_frames_from_video(
    video_path: str,
    start_frame: int,
    num_frames: int,
) -> List[np.ndarray]:
    """
    Extract frames from a video.
    
    Args:
        video_path: Path to video file
        start_frame: Starting frame index (0-based)
        num_frames: Number of frames to extract
        
    Returns:
        List of frames as numpy arrays (BGR format)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Clamp to valid range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = min(total_frames, start_frame + num_frames)
    
    frames = []
    frame_idx = 0
    
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx >= start_frame and frame_idx < end_frame:
            frames.append(frame)
            
        frame_idx += 1
        
        if frame_idx >= end_frame:
            break
    
    cap.release()
    
    # Pad if needed
    if len(frames) < num_frames and len(frames) > 0:
        last_frame = frames[-1]
        frames.extend([last_frame] * (num_frames - len(frames)))
    
    return frames


def save_video_clip(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 24,
):
    """
    Save frames as a video file.
    
    Args:
        frames: List of frames as numpy arrays (BGR format)
        output_path: Path to save video
        fps: Frames per second
    """
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape[:2]
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()


def compute_metrics_with_ffmpeg(
    reference_video: str,
    distorted_video: str,
    width: int,
    height: int,
) -> Dict[str, float]:
    """
    Compute SSIM, PSNR, and VMAF scores between two videos using ffmpeg.
    
    Args:
        reference_video: Path to reference (ground truth) video
        distorted_video: Path to distorted (generated) video
        width: Video width
        height: Video height
        
    Returns:
        Dictionary with 'ssim', 'psnr', and 'vmaf' scores
    """
    metrics = {
        "ssim": 0.0,
        "psnr": 0.0,
        "vmaf": 0.0,
    }
    
    # Check if input videos exist
    if not os.path.exists(reference_video):
        print(f"    Warning: Reference video not found: {reference_video}")
        return metrics
    if not os.path.exists(distorted_video):
        print(f"    Warning: Distorted video not found: {distorted_video}")
        return metrics
    
    try:
        # Use custom FFmpeg with VMAF support
        ffmpeg_bin = "/workspace/deepvidsumm/ffmpeg/ffmpeg-n8.0-latest-linux64-gpl-8.0/bin/ffmpeg"
        
        # Create unique temporary log files based on process ID
        import tempfile
        temp_dir = tempfile.gettempdir()
        pid = os.getpid()
        ssim_log = os.path.join(temp_dir, f"ssim_log_{pid}.txt")
        psnr_log = os.path.join(temp_dir, f"psnr_log_{pid}.txt")
        vmaf_log = os.path.join(temp_dir, f"vmaf_log_{pid}.json")
        
        # Remove old log files if they exist
        for log_file in [ssim_log, psnr_log, vmaf_log]:
            if os.path.exists(log_file):
                os.remove(log_file)
        
        # Compute SSIM and PSNR together
        # Note: Both inputs need to be scaled to the same resolution first
        cmd_ssim_psnr = [
            ffmpeg_bin,
            "-y",  # Overwrite output files
            "-i", distorted_video,
            "-i", reference_video,
            "-lavfi",
            f"[0:v]scale={width}:{height}:flags=bicubic[dist];"
            f"[1:v]scale={width}:{height}:flags=bicubic[ref];"
            f"[dist][ref]ssim=stats_file={ssim_log}",
            "-f", "null",
            "-"
        ]
        
        print(f"    Running SSIM computation...")
        result = subprocess.run(
            cmd_ssim_psnr,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"    Warning: SSIM computation failed with return code {result.returncode}")
            stderr_output = result.stderr.decode('utf-8', errors='ignore')
            # Find the actual error in the stderr (usually near the end)
            stderr_lines = stderr_output.split('\n')
            error_lines = [line for line in stderr_lines if 'error' in line.lower() or 'invalid' in line.lower() or 'failed' in line.lower()]
            if error_lines:
                print(f"    FFmpeg errors: {' | '.join(error_lines[-5:])}")
            else:
                # Show last few lines of stderr which usually contain the error
                print(f"    FFmpeg stderr (last 10 lines):")
                for line in stderr_lines[-10:]:
                    if line.strip():
                        print(f"      {line}")
        
        # Now compute PSNR separately
        cmd_psnr = [
            ffmpeg_bin,
            "-y",
            "-i", distorted_video,
            "-i", reference_video,
            "-lavfi",
            f"[0:v]scale={width}:{height}:flags=bicubic[dist];"
            f"[1:v]scale={width}:{height}:flags=bicubic[ref];"
            f"[dist][ref]psnr=stats_file={psnr_log}",
            "-f", "null",
            "-"
        ]
        
        print(f"    Running PSNR computation...")
        result = subprocess.run(
            cmd_psnr,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"    Warning: PSNR computation failed with return code {result.returncode}")
            stderr_output = result.stderr.decode('utf-8', errors='ignore')
            stderr_lines = stderr_output.split('\n')
            error_lines = [line for line in stderr_lines if 'error' in line.lower() or 'invalid' in line.lower() or 'failed' in line.lower()]
            if error_lines:
                print(f"    FFmpeg errors: {' | '.join(error_lines[-5:])}")
            else:
                print(f"    FFmpeg stderr (last 10 lines):")
                for line in stderr_lines[-10:]:
                    if line.strip():
                        print(f"      {line}")
        
        # Parse SSIM from log file
        if os.path.exists(ssim_log):
            with open(ssim_log, 'r') as f:
                lines = f.readlines()
                if lines:
                    # SSIM log format: n:1 Y:0.95 U:0.96 V:0.97 All:0.96 (dB)
                    ssim_values = []
                    for line in lines:
                        parts = line.strip().split()
                        for part in parts:
                            if part.startswith("All:"):
                                try:
                                    ssim_val = float(part.split(":")[1])
                                    ssim_values.append(ssim_val)
                                except:
                                    pass
                    if ssim_values:
                        metrics["ssim"] = np.mean(ssim_values)
                        print(f"    Parsed {len(ssim_values)} SSIM values, mean: {metrics['ssim']:.4f}")
            os.remove(ssim_log)
        else:
            print(f"    Warning: SSIM log file not created: {ssim_log}")
        
        # Parse PSNR from log file
        if os.path.exists(psnr_log):
            with open(psnr_log, 'r') as f:
                lines = f.readlines()
                if lines:
                    # PSNR log format: n:1 mse_avg:123.45 mse_y:120.00 mse_u:125.00 mse_v:125.00 psnr_avg:28.50 psnr_y:29.00 psnr_u:28.00 psnr_v:28.00
                    psnr_values = []
                    for line in lines:
                        parts = line.strip().split()
                        for part in parts:
                            if part.startswith("psnr_avg:"):
                                try:
                                    psnr_val = float(part.split(":")[1])
                                    psnr_values.append(psnr_val)
                                except:
                                    pass
                    if psnr_values:
                        metrics["psnr"] = np.mean(psnr_values)
                        print(f"    Parsed {len(psnr_values)} PSNR values, mean: {metrics['psnr']:.2f} dB")
            os.remove(psnr_log)
        else:
            print(f"    Warning: PSNR log file not created: {psnr_log}")
        
        # Compute VMAF
        cmd_vmaf = [
            ffmpeg_bin,
            "-y",
            "-i", distorted_video,
            "-i", reference_video,
            "-lavfi",
            f"[0:v]scale={width}:{height}:flags=bicubic[dist];"
            f"[1:v]scale={width}:{height}:flags=bicubic[ref];"
            f"[dist][ref]libvmaf=log_fmt=json:log_path={vmaf_log}",
            "-f", "null",
            "-"
        ]
        
        print(f"    Running VMAF computation...")
        result = subprocess.run(
            cmd_vmaf,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"    Warning: VMAF computation failed with return code {result.returncode}")
            stderr_output = result.stderr.decode('utf-8', errors='ignore')
            stderr_lines = stderr_output.split('\n')
            error_lines = [line for line in stderr_lines if 'error' in line.lower() or 'invalid' in line.lower() or 'failed' in line.lower()]
            if error_lines:
                print(f"    FFmpeg errors: {' | '.join(error_lines[-5:])}")
        
        # Parse VMAF from JSON log
        if os.path.exists(vmaf_log):
            with open(vmaf_log, 'r') as f:
                vmaf_data = json.load(f)
                vmaf_score = vmaf_data.get("pooled_metrics", {}).get("vmaf", {}).get("mean", 0.0)
                metrics["vmaf"] = vmaf_score
                print(f"    Parsed VMAF score: {metrics['vmaf']:.2f}")
            os.remove(vmaf_log)
        else:
            print(f"    Warning: VMAF log file not created: {vmaf_log}")
        
    except subprocess.TimeoutExpired:
        print(f"    Warning: Metric computation timed out")
    except FileNotFoundError:
        print(f"    Warning: ffmpeg not found. Please install ffmpeg.")
    except Exception as e:
        print(f"    Warning: Metric computation failed: {e}")
    
    return metrics


def evaluate_single_cut(
    video_path: str,
    cut_frame: int,
    start_duration: int,
    mid_duration: int,
    end_duration: int,
    base_model_path: str,
    lora_path: str,
    output_dir: str,
    video_name: str,
    cut_idx: int,
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 50,
    out_fps: int = 24,
    transformer_precision: str = "bf16",
    vae_precision: str = "fp32",
    accelerator: Accelerator = None,
    # Bidirectional model parameters
    lora_fwd_path: Optional[str] = None,
    lora_bwd_path: Optional[str] = None,
    fusion_mlp_path: Optional[str] = None,
    attn_implementation: str = "sdpa",
    fusion_hidden_dim: int = 256,
    fusion_num_layers: int = 3,
    cnn_feature_dim: int = 64,
) -> Dict:
    """
    Evaluate model on a single cut.
    
    Args:
        video_path: Path to video file
        cut_frame: Frame index of the cut
        start_duration: Number of frames before cut to use as conditioning
        mid_duration: Number of frames to generate (ground truth exists)
        end_duration: Number of frames after cut to use as conditioning
        base_model_path: Path to base model
        lora_path: Path to LoRA weights (for unidirectional model)
        output_dir: Directory to save outputs
        video_name: Name of the video (for output files)
        cut_idx: Index of this cut in the video
        height: Output height
        width: Output width
        num_inference_steps: Number of inference steps
        out_fps: Output FPS
        transformer_precision: Transformer precision
        vae_precision: VAE precision
        lora_fwd_path: Path to forward LoRA weights (for bidirectional model)
        lora_bwd_path: Path to backward LoRA weights (for bidirectional model)
        fusion_mlp_path: Path to fusion MLP weights (if using bidirectional model)
        attn_implementation: Attention implementation ("sdpa", "flash_attention_2", "eager")
        fusion_hidden_dim: Hidden dim of fusion MLP (if bidirectional)
        fusion_num_layers: Num layers in fusion MLP (if bidirectional)
        cnn_feature_dim: CNN feature dim in fusion MLP (if bidirectional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Determine if using bidirectional model (requires both LoRAs and fusion MLP)
    use_bidirectional = (lora_fwd_path is not None and lora_bwd_path is not None 
                         and fusion_mlp_path is not None)
    
    # Create output subdirectory for this cut
    cut_output_dir = os.path.join(output_dir, f"{video_name}_cut{cut_idx}")
    os.makedirs(cut_output_dir, exist_ok=True)
    
    try:
        # Extract ground truth middle frames
        gt_start = cut_frame - mid_duration // 2
        gt_frames = extract_frames_from_video(video_path, gt_start, mid_duration)
        
        if len(gt_frames) == 0:
            print(f"  Warning: No ground truth frames extracted")
            return None
        
        # Resize ground truth frames to match the target resolution
        # This ensures fair comparison with generated frames
        gt_frames_resized = []
        for frame in gt_frames:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            gt_frames_resized.append(resized)
        
        # Save ground truth clip (at target resolution)
        gt_video_path = os.path.join(cut_output_dir, "ground_truth.mp4")
        save_video_clip(gt_frames_resized, gt_video_path, out_fps)
        
        # Validate ground truth video was created
        if not os.path.exists(gt_video_path):
            print(f"  Error: Failed to save ground truth video")
            return None
        
        # Check video properties
        cap = cv2.VideoCapture(gt_video_path)
        if cap.isOpened():
            gt_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            gt_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            gt_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"    Ground truth video: {gt_frame_count} frames, {gt_width}x{gt_height}")
        else:
            print(f"  Error: Ground truth video cannot be opened")
            return None
        
        # Extract start and end segments for conditioning
        start_frame_idx = cut_frame - mid_duration // 2 - start_duration
        end_frame_idx = cut_frame + mid_duration // 2
        
        # Generate inbetween frames
        generated_video_path = os.path.join(cut_output_dir, "generated_full.mp4")
        
        if use_bidirectional:
            # Use bidirectional model with separate forward/backward LoRAs
            # Also generate unfused outputs for inspection
            result = generate_bidirectional_inbetween_from_two_videos(
                base_model_path=base_model_path,
                lora_fwd_path=lora_fwd_path,
                lora_bwd_path=lora_bwd_path,
                fusion_mlp_path=fusion_mlp_path,
                start_video_path=video_path,
                start_frame_index=max(0, start_frame_idx),
                start_duration=start_duration,
                end_video_path=video_path,
                end_frame_index=max(0, end_frame_idx),
                end_duration=end_duration,
                mid_frames=mid_duration,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                out_fps=out_fps,
                transformer_precision=transformer_precision,
                vae_precision=vae_precision,
                attn_implementation=attn_implementation,
                fusion_hidden_dim=fusion_hidden_dim,
                fusion_num_layers=fusion_num_layers,
                cnn_feature_dim=cnn_feature_dim,
                output_path=generated_video_path,
                generate_unfused=True,
            )
            # Result is a dict with 'fused', 'forward_only', 'backward_only' paths
            fwd_only_path = result.get('forward_only')
            bwd_only_path = result.get('backward_only')
        else:
            # Use standard unidirectional model
            fwd_only_path = None
            bwd_only_path = None
            generate_inbetween_from_two_videos(
                base_model_path=base_model_path,
                lora_path=lora_path,
                start_video_path=video_path,
                start_frame_index=max(0, start_frame_idx),
                start_duration=start_duration,
                end_video_path=video_path,
                end_frame_index=max(0, end_frame_idx),
                end_duration=end_duration,
                mid_frames=mid_duration,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                out_fps=out_fps,
                transformer_precision=transformer_precision,
                vae_precision=vae_precision,
                output_path=generated_video_path,
            )
        
        # Extract only the middle generated frames (skip conditioning frames)
        generated_all_frames = extract_frames_from_video(
            generated_video_path, 
            start_duration,  # Skip start conditioning frames
            mid_duration
        )
        
        if len(generated_all_frames) == 0:
            print(f"  Warning: No generated frames extracted")
            return None
        
        # Save generated middle frames only
        generated_mid_path = os.path.join(cut_output_dir, "generated_middle.mp4")
        save_video_clip(generated_all_frames, generated_mid_path, out_fps)
        
        # Validate generated video
        if not os.path.exists(generated_mid_path):
            print(f"  Error: Failed to save generated middle video")
            return None
            
        cap = cv2.VideoCapture(generated_mid_path)
        if cap.isOpened():
            gen_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            gen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            gen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"    Generated video: {gen_frame_count} frames, {gen_width}x{gen_height}")
        else:
            print(f"  Error: Generated video cannot be opened")
            return None
        
        num_frames = min(len(gt_frames_resized), len(generated_all_frames))
        
        # Compute all metrics using ffmpeg
        # Note: Videos are already at the same resolution, but we still specify it for ffmpeg
        print(f"    Computing metrics with ffmpeg...")
        metrics = compute_metrics_with_ffmpeg(
            gt_video_path,
            generated_mid_path,
            width,  # Both videos are already at this resolution
            height
        )
        
        # For bidirectional model, also compute metrics on unfused outputs
        metrics_fwd_only = None
        metrics_bwd_only = None
        if use_bidirectional and fwd_only_path and bwd_only_path:
            # Extract middle frames from forward-only output
            fwd_only_all_frames = extract_frames_from_video(
                fwd_only_path, start_duration, mid_duration
            )
            if len(fwd_only_all_frames) > 0:
                fwd_only_mid_path = os.path.join(cut_output_dir, "generated_middle_forward_only.mp4")
                save_video_clip(fwd_only_all_frames, fwd_only_mid_path, out_fps)
                print(f"    Computing forward-only metrics...")
                metrics_fwd_only = compute_metrics_with_ffmpeg(
                    gt_video_path, fwd_only_mid_path, width, height
                )
            
            # Extract middle frames from backward-only output
            bwd_only_all_frames = extract_frames_from_video(
                bwd_only_path, start_duration, mid_duration
            )
            if len(bwd_only_all_frames) > 0:
                bwd_only_mid_path = os.path.join(cut_output_dir, "generated_middle_backward_only.mp4")
                save_video_clip(bwd_only_all_frames, bwd_only_mid_path, out_fps)
                print(f"    Computing backward-only metrics...")
                metrics_bwd_only = compute_metrics_with_ffmpeg(
                    gt_video_path, bwd_only_mid_path, width, height
                )
        
        # Save conditioning frames for reference
        start_frames = extract_frames_from_video(
            video_path,
            max(0, start_frame_idx),
            start_duration
        )
        end_frames = extract_frames_from_video(
            video_path,
            max(0, end_frame_idx),
            end_duration
        )
        
        save_video_clip(
            start_frames,
            os.path.join(cut_output_dir, "start_conditioning.mp4"),
            out_fps
        )
        save_video_clip(
            end_frames,
            os.path.join(cut_output_dir, "end_conditioning.mp4"),
            out_fps
        )
        
        # Return metrics
        result = {
            "video_name": video_name,
            "cut_idx": cut_idx,
            "cut_frame": cut_frame,
            "num_frames": num_frames,
            "ssim": metrics["ssim"],
            "psnr": metrics["psnr"],
            "vmaf": metrics["vmaf"],
            "output_dir": cut_output_dir,
        }
        
        # Add unfused metrics if available
        if metrics_fwd_only:
            result["ssim_fwd_only"] = metrics_fwd_only["ssim"]
            result["psnr_fwd_only"] = metrics_fwd_only["psnr"]
            result["vmaf_fwd_only"] = metrics_fwd_only["vmaf"]
        if metrics_bwd_only:
            result["ssim_bwd_only"] = metrics_bwd_only["ssim"]
            result["psnr_bwd_only"] = metrics_bwd_only["psnr"]
            result["vmaf_bwd_only"] = metrics_bwd_only["vmaf"]
        
        return result
        
    except Exception as e:
        print(f"  Error evaluating cut {cut_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate inbetweening model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_inbetween_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--cut_annotations",
        type=str,
        required=True,
        help="Path to JSON file with cut annotations",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing videos",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to trained LoRA weights (for unidirectional model)",
    )
    parser.add_argument(
        "--lora_fwd_path",
        type=str,
        default=None,
        help="Path to forward LoRA weights (for bidirectional model)",
    )
    parser.add_argument(
        "--lora_bwd_path",
        type=str,
        default=None,
        help="Path to backward LoRA weights (for bidirectional model)",
    )
    parser.add_argument(
        "--fusion_mlp_path",
        type=str,
        default=None,
        help="Path to fusion MLP weights (for bidirectional model)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--start_duration",
        type=int,
        default=30,
        help="Number of conditioning frames before cut",
    )
    parser.add_argument(
        "--mid_duration",
        type=int,
        default=60,
        help="Number of middle frames to generate/evaluate",
    )
    parser.add_argument(
        "--end_duration",
        type=int,
        default=30,
        help="Number of conditioning frames after cut",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to evaluate (for testing)",
    )
    parser.add_argument(
        "--max_cuts_per_video",
        type=int,
        default=None,
        help="Maximum number of cuts per video to evaluate",
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
    
    # Check if using bidirectional mode (requires all three paths)
    use_bidirectional = (args.lora_fwd_path is not None and 
                         args.lora_bwd_path is not None and 
                         args.fusion_mlp_path is not None)
    
    # Validate: either unidirectional (lora_path) or bidirectional (fwd+bwd+fusion) required
    if use_bidirectional:
        # Bidirectional mode - don't need lora_path
        pass
    elif args.lora_path is not None:
        # Unidirectional mode
        pass
    else:
        parser.error("Either --lora_path (for unidirectional) OR "
                     "--lora_fwd_path + --lora_bwd_path + --fusion_mlp_path (for bidirectional) is required")
    
    # Initialize accelerator for multi-GPU support
    accelerator = Accelerator()
    
    # Load config
    cfg = load_config(args.config)
    
    # Get model parameters from config
    base_model_path = cfg.get("base_model_path")
    inference_cfg = cfg.get("inference", {})
    height = inference_cfg.get("height", 480)
    width = inference_cfg.get("width", 832)
    num_inference_steps = inference_cfg.get("num_inference_steps", 50)
    out_fps = inference_cfg.get("out_fps", 24)
    transformer_precision = cfg.get("transformer_precision", "bf16")
    vae_precision = cfg.get("vae_precision", "fp32")
    
    # Load cut annotations (only on main process)
    if accelerator.is_main_process:
        print(f"Loading cut annotations from: {args.cut_annotations}")
        if use_bidirectional:
            print(f"Using BIDIRECTIONAL model:")
            print(f"  Forward LoRA: {args.lora_fwd_path}")
            print(f"  Backward LoRA: {args.lora_bwd_path}")
            print(f"  Fusion MLP: {args.fusion_mlp_path}")
        else:
            print(f"Using STANDARD (unidirectional) model: {args.lora_path}")
    
    with open(args.cut_annotations, 'r') as f:
        cut_annotations = json.load(f)
    
    if accelerator.is_main_process:
        print(f"Found {len(cut_annotations)} videos with cuts")
        print(f"Using {accelerator.num_processes} GPU(s)")
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Wait for main process to create directory
    accelerator.wait_for_everyone()
    
    # Split work across GPUs
    video_items = list(cut_annotations.items())
    
    # Distribute videos across processes
    videos_per_process = len(video_items) // accelerator.num_processes
    remainder = len(video_items) % accelerator.num_processes
    
    start_idx = accelerator.process_index * videos_per_process + min(accelerator.process_index, remainder)
    if accelerator.process_index < remainder:
        videos_per_process += 1
    end_idx = start_idx + videos_per_process
    
    my_videos = video_items[start_idx:end_idx]
    
    if accelerator.is_main_process:
        print(f"\nDistributing {len(video_items)} videos across {accelerator.num_processes} processes")
    
    # Evaluate videos assigned to this process
    all_results = []
    video_count = 0
    
    for video_name, cut_frames in tqdm(
        my_videos, 
        desc=f"Videos (GPU {accelerator.process_index})",
        disable=not accelerator.is_local_main_process
    ):
        if args.max_videos and video_count >= args.max_videos:
            break
        
        video_count += 1
        
        # Find video file
        video_path = os.path.join(args.data_root, video_name)
        if not os.path.exists(video_path):
            if accelerator.is_local_main_process:
                print(f"Warning: Video not found: {video_path}")
            continue
        
        if accelerator.is_local_main_process:
            print(f"\n[GPU {accelerator.process_index}] Evaluating {video_name} ({len(cut_frames)} cuts)")
        
        # Evaluate each cut
        cut_count = 0
        for cut_idx, cut_frame in enumerate(cut_frames):
            if args.max_cuts_per_video and cut_count >= args.max_cuts_per_video:
                break
            
            cut_count += 1
            
            if accelerator.is_local_main_process:
                print(f"  Cut {cut_idx + 1}/{len(cut_frames)} at frame {cut_frame}")
            
            result = evaluate_single_cut(
                video_path=video_path,
                cut_frame=cut_frame,
                start_duration=args.start_duration,
                mid_duration=args.mid_duration,
                end_duration=args.end_duration,
                base_model_path=base_model_path,
                lora_path=args.lora_path,
                output_dir=args.output_dir,
                video_name=video_name.replace('.mp4', ''),
                cut_idx=cut_idx,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                out_fps=out_fps,
                transformer_precision=transformer_precision,
                vae_precision=vae_precision,
                accelerator=accelerator,
                # Bidirectional model parameters
                lora_fwd_path=args.lora_fwd_path,
                lora_bwd_path=args.lora_bwd_path,
                fusion_mlp_path=args.fusion_mlp_path,
                attn_implementation=args.attn_implementation,
                fusion_hidden_dim=args.fusion_hidden_dim,
                fusion_num_layers=args.fusion_num_layers,
                cnn_feature_dim=args.cnn_feature_dim,
            )
            
            if result:
                all_results.append(result)
                if accelerator.is_local_main_process:
                    print(f"    SSIM: {result['ssim']:.4f}")
                    print(f"    PSNR: {result['psnr']:.2f} dB")
                    print(f"    VMAF: {result['vmaf']:.2f}")
    
    # Gather results from all processes
    accelerator.wait_for_everyone()
    
    # Collect all results on main process
    all_results_gathered = all_results if accelerator.num_processes == 1 else accelerator.gather_for_metrics(all_results)
    
    # Save results to CSV (only on main process)
    if accelerator.is_main_process:
        if all_results_gathered:
            df = pd.DataFrame(all_results_gathered)
            csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"\n{'='*80}")
            print(f"Saved detailed results to: {csv_path}")
            
            # Print summary statistics
            print(f"\n{'='*80}")
            print("SUMMARY STATISTICS")
            print(f"{'='*80}")
            print(f"Total evaluated: {len(all_results_gathered)} cuts")
            print(f"\nSSIM: {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}")
            print(f"PSNR: {df['psnr'].mean():.2f} ± {df['psnr'].std():.2f} dB")
            print(f"VMAF: {df['vmaf'].mean():.2f} ± {df['vmaf'].std():.2f}")
            
            # Save summary
            summary = {
                "total_cuts": len(all_results_gathered),
                "ssim_mean": float(df['ssim'].mean()),
                "ssim_std": float(df['ssim'].std()),
                "psnr_mean": float(df['psnr'].mean()),
                "psnr_std": float(df['psnr'].std()),
                "vmaf_mean": float(df['vmaf'].mean()),
                "vmaf_std": float(df['vmaf'].std()),
            }
            
            summary_path = os.path.join(args.output_dir, "summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSaved summary to: {summary_path}")
        else:
            print("\nNo results to save")


if __name__ == "__main__":
    main()
