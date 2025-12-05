#!/bin/bash
# Evaluation script for inbetweening model
# Usage: ./run_evaluation.sh
#
# Supports both:
# - Standard (unidirectional) model: set LORA_PATH only
# - Bidirectional model: set LORA_PATH and FUSION_MLP_PATH

# Configuration
CONFIG="config/default_inbetween_config.yaml"
CUT_ANNOTATIONS="/workspace/deepvidsumm/dep/clipshots/annotations/train_middle_frames.json"
DATA_ROOT="/workspace/deepvidsumm/dep/clipshots/videos/ClipShots/videos/train"

# ============================================
# Model paths - Choose one of the following:
# ============================================

# Option 1: Standard (unidirectional) model
# LORA_PATH="/workspace/deepvidsumm/wan-vid-inbetween/outputs/inbetween_lora/"
# FUSION_MLP_PATH=""  # Leave empty for unidirectional model

# Option 2: Bidirectional model (uncomment to use)
LORA_PATH="/workspace/deepvidsumm/wan-vid-inbetween/outputs/bidirectional_inbetween_lora/checkpoints_bidirectional/checkpoint-1400/lora/"
FUSION_MLP_PATH="/workspace/deepvidsumm/wan-vid-inbetween/outputs/fusion_mlp/fusion_mlp.pt"

OUTPUT_DIR="outputs/evaluation"

# Evaluation parameters
START_DURATION=30  # Number of conditioning frames before cut
MID_DURATION=60    # Number of frames to generate/evaluate
END_DURATION=30    # Number of conditioning frames after cut

# Optional: Limit evaluation for testing
MAX_VIDEOS=1           # Set to empty string "" to evaluate all videos
MAX_CUTS_PER_VIDEO=2   # Set to empty string "" to evaluate all cuts

# Bidirectional model hyperparameters (if using bidirectional mode)
ATTN_IMPLEMENTATION="sdpa"  # "sdpa", "flash_attention_2", or "eager"
FUSION_HIDDEN_DIM=256
FUSION_NUM_LAYERS=3
CNN_FEATURE_DIM=64

# Build command arguments
CMD="--config $CONFIG \
    --cut_annotations $CUT_ANNOTATIONS \
    --data_root $DATA_ROOT \
    --lora_path $LORA_PATH \
    --output_dir $OUTPUT_DIR \
    --start_duration $START_DURATION \
    --mid_duration $MID_DURATION \
    --end_duration $END_DURATION \
    --attn_implementation $ATTN_IMPLEMENTATION"

# Add optional parameters if set
if [ -n "$MAX_VIDEOS" ]; then
    CMD="$CMD --max_videos $MAX_VIDEOS"
fi

if [ -n "$MAX_CUTS_PER_VIDEO" ]; then
    CMD="$CMD --max_cuts_per_video $MAX_CUTS_PER_VIDEO"
fi

# Add bidirectional parameters if fusion MLP path is set
if [ -n "$FUSION_MLP_PATH" ]; then
    CMD="$CMD --fusion_mlp_path $FUSION_MLP_PATH"
    CMD="$CMD --fusion_hidden_dim $FUSION_HIDDEN_DIM"
    CMD="$CMD --fusion_num_layers $FUSION_NUM_LAYERS"
    CMD="$CMD --cnn_feature_dim $CNN_FEATURE_DIM"
    echo "Using BIDIRECTIONAL model"
else
    echo "Using STANDARD (unidirectional) model"
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
