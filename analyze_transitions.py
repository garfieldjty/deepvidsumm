#!/usr/bin/env python3
"""
Analyze transition data from ClipShots annotations.
Outputs:
- Number of videos
- Distribution of number of transitions per video
- Distribution of transition gaps (time between transitions)
"""

import json
from collections import Counter
import statistics


def analyze_transitions(json_path):
    """Analyze transition data from JSON file."""
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Export filtered transitions to JSON with middle frame numbers
    import json as json_module
    json_output_path = json_path.replace('.json', '_middle_frames.json')
    middle_frames_data = {}
    
    for video_name, video_data in data.items():
        transitions = video_data['transitions']
        frame_num = int(video_data.get('frame_num', 0))
        # Filter transitions in valid range
        valid_transitions = [t for t in transitions if t[0] > 120 and t[1] < frame_num - 120 and t[1] - t[0] < 60]
        
        # Calculate middle frame for each valid transition
        middle_frames = []
        for start, end in valid_transitions:
            middle_frame = (start + end) // 2
            middle_frames.append(middle_frame)
        
        if middle_frames:  # Only add videos that have valid transitions
            middle_frames_data[video_name] = middle_frames
    
    # Write JSON file
    with open(json_output_path, 'w') as jsonfile:
        json_module.dump(middle_frames_data, jsonfile, indent=2)
    
    print(f"Middle frames JSON saved to: {json_output_path}")
    print(f"Total videos with valid transitions: {len(middle_frames_data)}")
    # Export filtered transitions to CSV
    import csv
    csv_path = json_path.replace('.json', '_transitions.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['video', 'prompt', 'start_frame_index', 'num_frames'])
        for video_name, video_data in data.items():
            transitions = video_data['transitions']
            frame_num = int(video_data.get('frame_num', 0))
            # Filter transitions in valid range
            valid_transitions = [t for t in transitions if t[0] > 60 and t[1] < frame_num - 60 and t[1] - t[0] < 40]
            n = len(valid_transitions)
            if n == 0:
                continue
            if n <= 3:
                sampled = valid_transitions
            else:
                # Evenly sample 3 transitions
                idxs = [int(i * (n - 1) / 2) for i in range(3)]
                sampled = [valid_transitions[i] for i in idxs]
            for start, end in sampled:
                clip_start = ((end + start) // 2) - 60
                writer.writerow([video_name, "", clip_start, 121])
    print(f"Filtered transitions saved to: {csv_path}")
    
    # Number of videos
    num_videos = len(data)
    print(f"Total number of videos: {num_videos}")
    print()
    
    # Analyze transitions per video
    transitions_per_video = []
    all_gaps = []
    
    for video_name, video_data in data.items():
        transitions = video_data['transitions']
        num_transitions = len(transitions)
        transitions_per_video.append(num_transitions)
        
        # Calculate transition gaps (end - start for each transition)
        for t in transitions:
            gap = t[1] - t[0]
            all_gaps.append(gap)
    
    # Distribution of number of transitions
    print("=" * 60)
    print("DISTRIBUTION OF NUMBER OF TRANSITIONS PER VIDEO")
    print("=" * 60)
    
    transitions_counter = Counter(transitions_per_video)
    print(f"Min transitions: {min(transitions_per_video)}")
    print(f"Max transitions: {max(transitions_per_video)}")
    print(f"Mean transitions: {statistics.mean(transitions_per_video):.2f}")
    print(f"Median transitions: {statistics.median(transitions_per_video):.2f}")
    print(f"Std dev: {statistics.stdev(transitions_per_video):.2f}")
    print()
    
    # Histogram of transitions per video (binned)
    print("Histogram (number of transitions ranges -> count of videos):")
    trans_bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, float('inf')]
    trans_labels = ["0", "1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100+"]
    for i in range(len(trans_bins) - 1):
        lower = trans_bins[i]
        upper = trans_bins[i + 1]
        count = sum(1 for t in transitions_per_video if lower <= t < upper)
        percentage = (count / num_videos) * 100
        bar = "█" * int(percentage)
        print(f"  {trans_labels[i]:>7s} transitions: {count:4d} ({percentage:5.2f}%) {bar}")
    print()
    
    # Distribution of transition gaps
    print("=" * 60)
    print("DISTRIBUTION OF TRANSITION GAPS (frames)")
    print("=" * 60)
    
    if all_gaps:
        print(f"Total gaps analyzed: {len(all_gaps)}")
        print(f"Min gap: {min(all_gaps)} frames")
        print(f"Max gap: {max(all_gaps)} frames")
        print(f"Mean gap: {statistics.mean(all_gaps):.2f} frames")
        print(f"Median gap: {statistics.median(all_gaps):.2f} frames")
        print(f"Std dev: {statistics.stdev(all_gaps):.2f} frames")
        print()
        
        # Histogram of gaps (binned)
        print("Histogram (gap ranges -> count):")
        bins = [0, 10, 20, 30, 50, 100, 200, 500, 1000, float('inf')]
        bin_labels = ["0-10", "10-20", "20-30", "30-50", "50-100", "100-200", "200-500", "500-1000", "1000+"]
        
        for i in range(len(bins) - 1):
            lower = bins[i]
            upper = bins[i + 1]
            count = sum(1 for gap in all_gaps if lower <= gap < upper)
            percentage = (count / len(all_gaps)) * 100
            bar = "█" * int(percentage)
            print(f"  {bin_labels[i]:>10s} frames: {count:6d} ({percentage:5.2f}%) {bar}")
    else:
        print("No gaps to analyze (all videos have 0 or 1 transitions)")
    
    print()


if __name__ == "__main__":
    json_path = "/home/tjiao/cv_proj/dep/clipshots/annotations/train.json"
    analyze_transitions(json_path)
