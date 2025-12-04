#!/bin/bash
# Evaluation script for inbetweening model
# Usage: ./run_evaluation.sh

# Configuration
CONFIG="config/default_inbetween_config.yaml"
CUT_ANNOTATIONS="/workspace/deepvidsumm/dep/clipshots/annotations/train_middle_frames.json"
DATA_ROOT="/workspace/deepvidsumm/dep/clipshots/videos/ClipShots/videos/train"
LORA_PATH="/workspace/deepvidsumm/wan-vid-inbetween/outputs/inbetween_lora/checkpoints/checkpoint-1000/lora"  # Path to trained LoRA weights (or use outputs/inbetween_lora/checkpoints/checkpoint-100)
OUTPUT_DIR="outputs/evaluation"

# Evaluation parameters
START_DURATION=30  # Number of conditioning frames before cut
MID_DURATION=60    # Number of frames to generate/evaluate
END_DURATION=30    # Number of conditioning frames after cut

# Optional: Limit evaluation for testing
MAX_VIDEOS=1           # Set to empty string "" to evaluate all videos
MAX_CUTS_PER_VIDEO=2   # Set to empty string "" to evaluate all cuts

# Build command arguments
CMD="--config $CONFIG \
    --cut_annotations $CUT_ANNOTATIONS \
    --data_root $DATA_ROOT \
    --lora_path $LORA_PATH \
    --output_dir $OUTPUT_DIR \
    --start_duration $START_DURATION \
    --mid_duration $MID_DURATION \
    --end_duration $END_DURATION"

# Add optional parameters if set
if [ -n "$MAX_VIDEOS" ]; then
    CMD="$CMD --max_videos $MAX_VIDEOS"
fi

if [ -n "$MAX_CUTS_PER_VIDEO" ]; then
    CMD="$CMD --max_cuts_per_video $MAX_CUTS_PER_VIDEO"
fi

# Run evaluation with accelerate (supports multi-GPU)
echo "Starting evaluation with accelerate..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Change to wan-vid-inbetween directory to use module imports
cd "$(dirname "$0")"
accelerate launch -m src.evaluate_inbetween $CMD

echo ""
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
