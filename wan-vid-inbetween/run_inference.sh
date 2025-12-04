#!/bin/bash
# Example script to run inference with accelerate
# Usage: ./run_inference.sh

# Configuration
CONFIG="config/default_inbetween_config.yaml"
LORA_PATH="outputs/inbetween_lora/final"   # Update this to your trained LoRA path

# Start video parameters
START_VIDEO_PATH="path/to/start_video.mp4" # Path to first video
START_FRAME_INDEX=0                         # Starting frame index in first video
START_DURATION=30                           # Number of frames to use from first video

# End video parameters
END_VIDEO_PATH="path/to/end_video.mp4"     # Path to second video
END_FRAME_INDEX=0                           # Starting frame index in second video
END_DURATION=30                             # Number of frames to use from second video

OUTPUT_PATH="outputs/inbetween_output.mp4"

# Run with accelerate (supports single GPU or multi-GPU)
accelerate launch -m src.run_inference_inbetween \
    --config "$CONFIG" \
    --lora_path "$LORA_PATH" \
    --start_video_path "$START_VIDEO_PATH" \
    --start_frame_index "$START_FRAME_INDEX" \
    --start_duration "$START_DURATION" \
    --end_video_path "$END_VIDEO_PATH" \
    --end_frame_index "$END_FRAME_INDEX" \
    --end_duration "$END_DURATION" \
    --output_path "$OUTPUT_PATH"

# Alternative: Run without accelerate (single GPU only)
# python -m src.run_inference_inbetween \
#     --config "$CONFIG" \
#     --lora_path "$LORA_PATH" \
#     --start_video_path "$START_VIDEO_PATH" \
#     --start_frame_index "$START_FRAME_INDEX" \
#     --start_duration "$START_DURATION" \
#     --end_video_path "$END_VIDEO_PATH" \
#     --end_frame_index "$END_FRAME_INDEX" \
#     --end_duration "$END_DURATION" \
#     --output_path "$OUTPUT_PATH"
