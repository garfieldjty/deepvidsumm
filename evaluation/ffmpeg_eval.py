#!/usr/bin/env python3
"""
Video Quality Evaluation Script
Calculates SSIM, PSNR, and VMAF scores for generated videos compared to original videos.
"""

import os
import subprocess
import re
import csv
import json
from typing import Dict, List, Tuple, Any
import argparse
import sys

# Path to ffmpeg binary
FFMPEG_PATH = "/workspace/ffmpeg-n8.0-latest-linux64-gpl-8.0/bin/ffmpeg"

def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run([FFMPEG_PATH, "-version"], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def extract_video_name(filename: str) -> str:
    """
    Extract base video name from filename.
    Original videos: {name}_ori.mp4
    Generated videos: {name}_inbetween.mp4
    """
    if filename.endswith("_ori.mp4"):
        return filename[:-8]  # Remove "_ori.mp4"
    elif filename.endswith("_inbetween.mp4"):
        return filename[:-14]
    else:
        return filename

def find_matching_videos(original_dir: str, generated_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching video pairs between original and generated directories.
    Returns list of (original_path, generated_path) tuples.
    """
    matches = []
    
    # Get all original videos
    original_files = {}
    for file in os.listdir(original_dir):
        if file.endswith(".mp4"):
            base_name = extract_video_name(file)
            original_files[base_name] = os.path.join(original_dir, file)
    
    # Get all generated videos
    generated_files = {}
    for file in os.listdir(generated_dir):
        if file.endswith(".mp4"):
            base_name = extract_video_name(file)
            generated_files[base_name] = os.path.join(generated_dir, file)
    
    # Find matches
    for base_name in original_files:
        if base_name in generated_files:
            matches.append((original_files[base_name], generated_files[base_name]))
        else:
            print(f"Warning: No matching generated video for original: {base_name}")
    
    return matches

def calculate_ssim(original_path: str, generated_path: str) -> Dict[str, float]:
    """
    Calculate SSIM (Structural Similarity Index) using ffmpeg.
    Returns dictionary with Y, U, V, and All components.
    Only scores frames 40-80.
    """
    cmd = [
        FFMPEG_PATH,
        "-i", original_path,
        "-i", generated_path,
        "-lavfi", "[0:v]trim=start_frame=40:end_frame=80,setpts=PTS-STARTPTS[ref];[1:v]trim=start_frame=40:end_frame=80,setpts=PTS-STARTPTS[dist];[ref][dist]ssim=stats_file=-",
        "-f", "null",
        "-"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stderr
        
        # Parse SSIM output - format: SSIM Y:0.650854 (4.569927) U:0.913816 (10.645711) V:0.930791 (11.598360) All:0.741337 (5.872656)
        ssim_pattern = r"SSIM Y:(\d+\.\d+) \([^)]+\) U:(\d+\.\d+) \([^)]+\) V:(\d+\.\d+) \([^)]+\) All:(\d+\.\d+)"
        match = re.search(ssim_pattern, output)
        
        if match:
            return {
                "ssim_y": float(match.group(1)),
                "ssim_u": float(match.group(2)),
                "ssim_v": float(match.group(3)),
                "ssim_all": float(match.group(4))
            }
        else:
            # Try legacy pattern (without dB values in parentheses)
            ssim_simple_pattern = r"SSIM Y:(\d+\.\d+) U:(\d+\.\d+) V:(\d+\.\d+) All:(\d+\.\d+)"
            match = re.search(ssim_simple_pattern, output)
            if match:
                return {
                    "ssim_y": float(match.group(1)),
                    "ssim_u": float(match.group(2)),
                    "ssim_v": float(match.group(3)),
                    "ssim_all": float(match.group(4))
                }
            else:
                # Try older format: All:0.741337 (Y:0.650854 U:0.913816 V:0.930791)
                ssim_old_pattern = r"All:(\d+\.\d+) \(Y:(\d+\.\d+) U:(\d+\.\d+) V:(\d+\.\d+)\)"
                match = re.search(ssim_old_pattern, output)
                if match:
                    return {
                        "ssim_all": float(match.group(1)),
                        "ssim_y": float(match.group(2)),
                        "ssim_u": float(match.group(3)),
                        "ssim_v": float(match.group(4))
                    }
                else:
                    print(f"Warning: Could not parse SSIM output for {os.path.basename(generated_path)}")
                    return {"ssim_all": 0.0, "ssim_y": 0.0, "ssim_u": 0.0, "ssim_v": 0.0}
                
    except subprocess.TimeoutExpired:
        print(f"Error: SSIM calculation timeout for {os.path.basename(generated_path)}")
        return {"ssim_all": 0.0, "ssim_y": 0.0, "ssim_u": 0.0, "ssim_v": 0.0}
    except Exception as e:
        print(f"Error calculating SSIM for {os.path.basename(generated_path)}: {e}")
        return {"ssim_all": 0.0, "ssim_y": 0.0, "ssim_u": 0.0, "ssim_v": 0.0}

def calculate_psnr(original_path: str, generated_path: str) -> Dict[str, float]:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) using ffmpeg.
    Returns dictionary with average, min, and max values.
    Only scores frames 40-80.
    """
    cmd = [
        FFMPEG_PATH,
        "-i", original_path,
        "-i", generated_path,
        "-lavfi", "[0:v]trim=start_frame=40:end_frame=80,setpts=PTS-STARTPTS[ref];[1:v]trim=start_frame=40:end_frame=80,setpts=PTS-STARTPTS[dist];[ref][dist]psnr=stats_file=-",
        "-f", "null",
        "-"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stderr
        
        # Parse PSNR output
        psnr_pattern = r"average:(\d+\.\d+) min:(\d+\.\d+) max:(\d+\.\d+)"
        match = re.search(psnr_pattern, output)
        
        if match:
            return {
                "psnr_avg": float(match.group(1)),
                "psnr_min": float(match.group(2)),
                "psnr_max": float(match.group(3))
            }
        else:
            # Try alternative pattern
            psnr_simple_pattern = r"PSNR y:(\d+\.\d+) u:(\d+\.\d+) v:(\d+\.\d+) average:(\d+\.\d+) min:(\d+\.\d+) max:(\d+\.\d+)"
            match = re.search(psnr_simple_pattern, output)
            if match:
                return {
                    "psnr_y": float(match.group(1)),
                    "psnr_u": float(match.group(2)),
                    "psnr_v": float(match.group(3)),
                    "psnr_avg": float(match.group(4)),
                    "psnr_min": float(match.group(5)),
                    "psnr_max": float(match.group(6))
                }
            else:
                print(f"Warning: Could not parse PSNR output for {os.path.basename(generated_path)}")
                return {"psnr_avg": 0.0, "psnr_min": 0.0, "psnr_max": 0.0}
                
    except subprocess.TimeoutExpired:
        print(f"Error: PSNR calculation timeout for {os.path.basename(generated_path)}")
        return {"psnr_avg": 0.0, "psnr_min": 0.0, "psnr_max": 0.0}
    except Exception as e:
        print(f"Error calculating PSNR for {os.path.basename(generated_path)}: {e}")
        return {"psnr_avg": 0.0, "psnr_min": 0.0, "psnr_max": 0.0}

def calculate_vmaf(original_path: str, generated_path: str) -> Dict[str, float]:
    """
    Calculate VMAF (Video Multi-method Assessment Fusion) using ffmpeg.
    Returns dictionary with VMAF score and optional motion score.
    Only scores frames 40-80.
    """
    cmd = [
        FFMPEG_PATH,
        "-i", original_path,
        "-i", generated_path,
        "-lavfi", "[0:v]trim=start_frame=40:end_frame=80,setpts=PTS-STARTPTS[ref];[1:v]trim=start_frame=40:end_frame=80,setpts=PTS-STARTPTS[dist];[ref][dist]libvmaf=log_fmt=json:log_path=/tmp/vmaf.json",
        "-f", "null",
        "-"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        # Try to read VMAF JSON output
        try:
            with open("/tmp/vmaf.json", "r") as f:
                vmaf_data = json.load(f)
            
            if "frames" in vmaf_data and len(vmaf_data["frames"]) > 0:
                # Calculate average VMAF score
                vmaf_scores = [frame["metrics"]["vmaf"] for frame in vmaf_data["frames"]]
                vmaf_avg = sum(vmaf_scores) / len(vmaf_scores)
                
                result_dict = {"vmaf_avg": vmaf_avg}
                
                # Check if motion score is available
                if "vmaf_motion" in vmaf_data["frames"][0]["metrics"]:
                    motion_scores = [frame["metrics"]["vmaf_motion"] for frame in vmaf_data["frames"]]
                    result_dict["vmaf_motion_avg"] = sum(motion_scores) / len(motion_scores)
                
                return result_dict
            else:
                print(f"Warning: No VMAF frames data for {os.path.basename(generated_path)}")
                return {"vmaf_avg": 0.0}
                
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse VMAF JSON for {os.path.basename(generated_path)}: {e}")
            
            # Try to parse from stderr as fallback
            output = result.stderr
            vmaf_pattern = r"VMAF score: (\d+\.\d+)"
            match = re.search(vmaf_pattern, output)
            if match:
                return {"vmaf_avg": float(match.group(1))}
            else:
                return {"vmaf_avg": 0.0}
                
    except subprocess.TimeoutExpired:
        print(f"Error: VMAF calculation timeout for {os.path.basename(generated_path)}")
        return {"vmaf_avg": 0.0}
    except Exception as e:
        print(f"Error calculating VMAF for {os.path.basename(generated_path)}: {e}")
        return {"vmaf_avg": 0.0}

def create_multi_model_comparison(video_name: str, original_path: str, model_paths: Dict[str, str], 
                                 output_path: str) -> bool:
    """
    Extract 6 frames (frames 40, 50, 60, 70, 80, 90) and create a vertical 
    concatenation comparing original vs all models side by side.
    Returns True if successful.
    """
    try:
        # Extract frames from all videos
        frame_nums = [40, 50, 60, 70, 80, 90]
        all_frames = {}
        
        # Extract original frames
        all_frames['original'] = []
        for frame_num in frame_nums:
            orig_frame = f"/tmp/orig_{video_name}_frame_{frame_num}.png"
            cmd = [
                FFMPEG_PATH,
                "-i", original_path,
                "-vf", f"select='eq(n\\,{frame_num})'",
                "-vframes", "1",
                "-y",
                orig_frame
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
            all_frames['original'].append(orig_frame)
        
        # Extract frames from each model
        for model_name, model_path in model_paths.items():
            all_frames[model_name] = []
            for frame_num in frame_nums:
                frame_file = f"/tmp/{model_name}_{video_name}_frame_{frame_num}.png"
                cmd = [
                    FFMPEG_PATH,
                    "-i", model_path,
                    "-vf", f"select='eq(n\\,{frame_num})'",
                    "-vframes", "1",
                    "-y",
                    frame_file
                ]
                subprocess.run(cmd, capture_output=True, timeout=30)
                all_frames[model_name].append(frame_file)
        
        # Create 4-column comparison for each frame (original + 3 models)
        comparison_rows = []
        for i, frame_num in enumerate(frame_nums):
            row_frame = f"/tmp/row_{video_name}_frame_{frame_num}.png"
            
            # Build filter complex for horizontal stacking with labels
            filter_parts = []
            filter_parts.append(f"[0:v]drawtext=text='Original F{frame_num}':fontsize=20:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.5[orig]")
            
            input_idx = 1
            labeled_streams = ['[orig]']
            for model_name in sorted(model_paths.keys()):
                filter_parts.append(f"[{input_idx}:v]drawtext=text='{model_name.title()} F{frame_num}':fontsize=20:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.5[{model_name}]")
                labeled_streams.append(f'[{model_name}]')
                input_idx += 1
            
            filter_parts.append(f"{''.join(labeled_streams)}hstack=inputs={len(labeled_streams)}")
            filter_complex = ";".join(filter_parts)
            
            # Build command with all inputs
            cmd = [FFMPEG_PATH, "-i", all_frames['original'][i]]
            for model_name in sorted(model_paths.keys()):
                cmd.extend(["-i", all_frames[model_name][i]])
            cmd.extend([
                "-filter_complex", filter_complex,
                "-y", row_frame
            ])
            
            subprocess.run(cmd, capture_output=True, timeout=30)
            comparison_rows.append(row_frame)
        
        # Stack all rows vertically
        cmd_vstack = [FFMPEG_PATH]
        for row in comparison_rows:
            cmd_vstack.extend(["-i", row])
        
        filter_inputs = "".join([f"[{i}:v]" for i in range(len(comparison_rows))])
        cmd_vstack.extend([
            "-filter_complex", f"{filter_inputs}vstack=inputs={len(comparison_rows)}",
            "-y", output_path
        ])
        subprocess.run(cmd_vstack, capture_output=True, timeout=30)
        
        # Clean up temporary files
        for frames_list in all_frames.values():
            for frame in frames_list:
                try:
                    os.remove(frame)
                except OSError:
                    pass
        for row in comparison_rows:
            try:
                os.remove(row)
            except OSError:
                pass
        
        return True
        
    except Exception as e:
        print(f"Error creating multi-model comparison: {e}")
        return False

def evaluate_video_pair(original_path: str, generated_path: str, model_type: str) -> Dict[str, Any]:
    """
    Evaluate a single video pair and return all metrics.
    """
    video_name = extract_video_name(os.path.basename(generated_path))
    
    print(f"Evaluating: {video_name} ({model_type})")
    
    results: Dict[str, Any] = {
        "video_name": video_name,
        "model_type": model_type,
        "original_path": original_path,
        "generated_path": generated_path
    }
    
    # Calculate SSIM
    print(f"  Calculating SSIM...")
    ssim_results = calculate_ssim(original_path, generated_path)
    results.update(ssim_results)
    
    # Calculate PSNR
    print(f"  Calculating PSNR...")
    psnr_results = calculate_psnr(original_path, generated_path)
    results.update(psnr_results)
    
    # Calculate VMAF
    print(f"  Calculating VMAF...")
    vmaf_results = calculate_vmaf(original_path, generated_path)
    results.update(vmaf_results)
    
    print(f"  Done. SSIM: {results.get('ssim_all', 0):.4f}, PSNR: {results.get('psnr_avg', 0):.2f}, VMAF: {results.get('vmaf_avg', 0):.2f}")
    
    return results

def save_results_to_csv(results: List[Dict], output_path: str):
    """
    Save evaluation results to CSV file.
    """
    if not results:
        print("No results to save")
        return
    
    # Extract all possible keys from results
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    
    # Define column order
    fieldnames = [
        "video_name", "model_type",
        "ssim_all", "ssim_y", "ssim_u", "ssim_v",
        "psnr_avg", "psnr_min", "psnr_max", "psnr_y", "psnr_u", "psnr_v",
        "vmaf_avg", "vmaf_motion_avg",
        "original_path", "generated_path"
    ]
    
    # Filter to only include fields that exist in fieldnames
    fieldnames = [f for f in fieldnames if f in all_keys]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Create row with only the fields we're writing
            row = {field: result.get(field, "") for field in fieldnames}
            writer.writerow(row)
    
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Calculate video quality metrics (SSIM, PSNR, VMAF)")
    parser.add_argument("--output", "-o", default="video_quality_results.csv",
                       help="Output CSV file path")
    parser.add_argument("--skip-ssim", action="store_true",
                       help="Skip SSIM calculation")
    parser.add_argument("--skip-psnr", action="store_true",
                       help="Skip PSNR calculation")
    parser.add_argument("--skip-vmaf", action="store_true",
                       help="Skip VMAF calculation")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: only process first 2 videos")
    parser.add_argument("--create-comparison", action="store_true",
                       help="Create frame comparison images (frames 40, 50, 60, 70, 80, 90)")
    parser.add_argument("--comparison-dir", default="frame_comparisons",
                       help="Directory to save frame comparison images")
    
    args = parser.parse_args()
    
    # Check ffmpeg
    if not check_ffmpeg():
        print(f"Error: ffmpeg not found at {FFMPEG_PATH}")
        print("Please ensure ffmpeg is installed and accessible")
        sys.exit(1)
    
    # Define directories
    base_dir = "/workspace/deepvidsumm/evaluation"
    original_dir = os.path.join(base_dir, "original_videos")
    
    # Create comparison directory if needed
    comparison_dir = None
    if args.create_comparison:
        comparison_dir = os.path.join(base_dir, args.comparison_dir)
        os.makedirs(comparison_dir, exist_ok=True)
        print(f"Frame comparisons will be saved to: {comparison_dir}")
    
    # Define model directories to evaluate
    model_dirs = [
        ("longcat_lora", os.path.join(base_dir, "longcat_lora_output")),
        ("longcat_vanilla", os.path.join(base_dir, "longcat_vanilla_output")),
        ("wan", os.path.join(base_dir, "wan_output"))
    ]
    
    all_results = []
    
    # First pass: collect all video pairs organized by video name
    video_map = {}  # video_name -> {model_name: (original_path, generated_path)}
    
    for model_name, generated_dir in model_dirs:
        if not os.path.exists(generated_dir):
            print(f"Warning: Directory {generated_dir} does not exist, skipping {model_name}")
            continue
        
        if not any(f.endswith(".mp4") for f in os.listdir(generated_dir)):
            print(f"Warning: No MP4 files found in {generated_dir}, skipping {model_name}")
            continue
        
        # Find matching video pairs
        video_pairs = find_matching_videos(original_dir, generated_dir)
        
        for original_path, generated_path in video_pairs:
            video_name = extract_video_name(os.path.basename(generated_path))
            if video_name not in video_map:
                video_map[video_name] = {}
            video_map[video_name][model_name] = (original_path, generated_path)
    
    # Limit videos in test mode
    if args.test:
        video_map = dict(list(video_map.items())[:2])
        print(f"Test mode: Processing first {len(video_map)} videos")
    
    # Second pass: evaluate and create comparisons
    for video_idx, (video_name, model_data) in enumerate(sorted(video_map.items()), 1):
        print(f"\n{'='*50}")
        print(f"Video {video_idx}/{len(video_map)}: {video_name}")
        print(f"{'='*50}")
        
        # Evaluate each model for this video
        for model_name in sorted(model_data.keys()):
            original_path, generated_path = model_data[model_name]
            
            # Skip metrics if requested
            if args.skip_ssim and args.skip_psnr and args.skip_vmaf:
                print(f"Skipping metrics for {model_name}")
                continue
            
            results = evaluate_video_pair(original_path, generated_path, model_name)
            all_results.append(results)
        
        # Create multi-model comparison if requested and we have multiple models
        if args.create_comparison and comparison_dir and len(model_data) >= 2:
            print("\nCreating multi-model comparison...")
            original_path = list(model_data.values())[0][0]
            
            model_paths = {model_name: generated_path 
                          for model_name, (_, generated_path) in model_data.items()}
            
            comparison_path = os.path.join(comparison_dir, f"{video_name}_all_models_comparison.png")
            if create_multi_model_comparison(video_name, original_path, model_paths, comparison_path):
                print(f"  Saved multi-model comparison to {comparison_path}")
    
    # Save results
    if all_results:
        save_results_to_csv(all_results, args.output)
        
        # Print summary
        print("\n" + "=" * 50)
        print("Evaluation Summary")
        print("=" * 50)
        
        for model_name in set(r["model_type"] for r in all_results):
            model_results = [r for r in all_results if r["model_type"] == model_name]
            
            if model_results:
                ssim_avg = sum(r.get("ssim_all", 0) for r in model_results) / len(model_results)
                psnr_avg = sum(r.get("psnr_avg", 0) for r in model_results) / len(model_results)
                vmaf_avg = sum(r.get("vmaf_avg", 0) for r in model_results) / len(model_results)
                
                print(f"{model_name}:")
                print(f"  Videos: {len(model_results)}")
                print(f"  Avg SSIM: {ssim_avg:.4f}")
                print(f"  Avg PSNR: {psnr_avg:.2f} dB")
                print(f"  Avg VMAF: {vmaf_avg:.2f}")
                print()
    else:
        print("No results generated")

if __name__ == "__main__":
    main()
