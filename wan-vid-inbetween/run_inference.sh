#!/bin/bash
# Example script to run inference with accelerate
# Usage: ./run_inference.sh
#
# Supports both:
# - Standard (unidirectional) model: set LORA_PATH only
# - Bidirectional model: set LORA_PATH and FUSION_MLP_PATH

# Configuration
CONFIG="config/default_inbetween_config.yaml"

# ============================================
# Model paths - Choose one of the following:
# ============================================

# Option 1: Standard (unidirectional) model
LORA_PATH="outputs/inbetween_lora/"
FUSION_MLP_PATH=""  # Leave empty for unidirectional model

# Option 2: Bidirectional model (uncomment to use)
# LORA_PATH="outputs/bidirectional_inbetween_lora/lora/"
# FUSION_MLP_PATH="outputs/bidirectional_inbetween_lora/fusion_mlp.pt"

# Start video parameters
START_VIDEO_PATH="path/to/start_video.mp4" # Path to first video
START_FRAME_INDEX=0                         # Starting frame index in first video
START_DURATION=30                           # Number of frames to use from first video

# End video parameters
END_VIDEO_PATH="path/to/end_video.mp4"     # Path to second video
END_FRAME_INDEX=0                           # Starting frame index in second video
END_DURATION=30                             # Number of frames to use from second video

OUTPUT_PATH="outputs/inbetween_output.mp4"

# Bidirectional model hyperparameters (if using bidirectional mode)
ATTN_IMPLEMENTATION="sdpa"  # "sdpa", "flash_attention_2", or "eager"
FUSION_HIDDEN_DIM=256
FUSION_NUM_LAYERS=3
CNN_FEATURE_DIM=64

# Build command
CMD="--config $CONFIG \
    --lora_path $LORA_PATH \
    --start_video_path $START_VIDEO_PATH \
    --start_frame_index $START_FRAME_INDEX \
    --start_duration $START_DURATION \
    --end_video_path $END_VIDEO_PATH \
    --end_frame_index $END_FRAME_INDEX \
    --end_duration $END_DURATION \
    --output_path $OUTPUT_PATH \
    --attn_implementation $ATTN_IMPLEMENTATION"

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

# Run with accelerate (supports single GPU or multi-GPU)
accelerate launch -m src.run_inference_inbetween $CMD

# Alternative: Run without accelerate (single GPU only)
# python -m src.run_inference_inbetween $CMD
