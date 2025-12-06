#!/usr/bin/env python3
"""
Generate a JSON annotations file for the InbetweenVideoDataset.

This script scans a video directory and creates a JSON file with video metadata
that can be used by the dataset. The format matches the expected cut annotations
format: { "video_name.mp4": [cut_frame_indices], ... }

Usage:
    python scripts/generate_video_annotations.py \
        --video_dir /path/to/videos \
        --output /path/to/output.json \
        --recursive

Options:
    --video_dir     Root directory containing videos
    --output        Output JSON file path
    --recursive     Search for videos recursively (default: True)
    --extensions    Video extensions to include (default: mp4,avi,mov,mkv)
    --include_frame_count  Include frame count info (slower but useful for validation)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from tqdm import tqdm


def get_video_frame_count(video_path: str) -> int:
    """Get the frame count of a video file."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def find_videos(
    video_dir: str,
    extensions: List[str],
    recursive: bool = True
) -> List[Path]:
    """Find all video files in the directory."""
    video_dir = Path(video_dir)
    videos = []
    
    for ext in extensions:
        pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
        videos.extend(video_dir.glob(pattern))
    
    return sorted(videos)


def generate_annotations(
    video_dir: str,
    output_path: str,
    extensions: List[str] = ["mp4", "avi", "mov", "mkv"],
    recursive: bool = True,
    include_frame_count: bool = False,
    min_frames: int = 0,
) -> Dict[str, List[int]]:
    """
    Generate annotations JSON file for the dataset.
    
    Args:
        video_dir: Root directory containing videos
        output_path: Path to save the JSON file
        extensions: List of video file extensions to include
        recursive: Whether to search recursively
        include_frame_count: If True, validates videos and filters by frame count
        min_frames: Minimum number of frames required (only used if include_frame_count=True)
    
    Returns:
        Dictionary mapping video filenames to cut frame indices
    """
    print(f"Scanning for videos in: {video_dir}")
    videos = find_videos(video_dir, extensions, recursive)
    print(f"Found {len(videos)} video files")
    
    annotations = {}
    skipped = 0
    
    for video_path in tqdm(videos, desc="Processing videos"):
        video_name = video_path.name
        
        if include_frame_count:
            try:
                frame_count = get_video_frame_count(str(video_path))
                if frame_count < min_frames:
                    skipped += 1
                    continue
                # Store empty list - no cut annotations
                # The dataset will use random sampling for these videos
                annotations[video_name] = []
            except Exception as e:
                print(f"Warning: Failed to read {video_path}: {e}")
                skipped += 1
                continue
        else:
            # Just add the video with empty cut list
            annotations[video_name] = []
    
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\nSaved annotations to: {output_path}")
    print(f"Total videos: {len(annotations)}")
    if skipped > 0:
        print(f"Skipped: {skipped}")
    
    return annotations


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON annotations file for InbetweenVideoDataset"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Root directory containing videos"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search for videos recursively (default: True)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Don't search recursively"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default="mp4,avi,mov,mkv",
        help="Comma-separated list of video extensions (default: mp4,avi,mov,mkv)"
    )
    parser.add_argument(
        "--include_frame_count",
        action="store_true",
        help="Validate videos and include frame count filtering (slower)"
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=120,
        help="Minimum frames required per video (default: 120, only with --include_frame_count)"
    )
    
    args = parser.parse_args()
    
    extensions = [ext.strip().lower() for ext in args.extensions.split(",")]
    
    generate_annotations(
        video_dir=args.video_dir,
        output_path=args.output,
        extensions=extensions,
        recursive=args.recursive,
        include_frame_count=args.include_frame_count,
        min_frames=args.min_frames,
    )


if __name__ == "__main__":
    main()
