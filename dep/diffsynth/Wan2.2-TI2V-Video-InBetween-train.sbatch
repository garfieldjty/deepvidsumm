#!/bin/bash
#SBATCH --job-name=wan22-ti2v-inbetween    # job name
#SBATCH --nodes=1                           # node count
#SBATCH --gpus=1                            # number of GPUs per node
#SBATCH --time=20:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --partition=normal                  # partition(large/normal/cpu)
#SBATCH --account=mscaisuperpod            # only required for multiple projects

# ============================================================================
# Wan2.2-TI2V-5B Video InBetweening Training Script
# 
# This script trains the Wan-AI/Wan2.2-TI2V-5B model for video inbetweening.
# The model learns to generate intermediate frames given start and end frames.
# ============================================================================

# Print GPU info for debugging
echo "=== GPU Information ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --list-gpus

# Performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512  # Reduce memory fragmentation
export CUDA_LAUNCH_BLOCKING=0  # Async CUDA operations
export TORCH_CUDNN_V8_API_ENABLED=1  # Use latest cuDNN
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Optimize for single GPU
export NCCL_ASYNC_ERROR_HANDLING=1  # Better error handling
export TORCH_NCCL_BLOCKING_WAIT=0  # Non-blocking NCCL operations
export OMP_NUM_THREADS=4  # OpenMP threads for data loading
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings

# Enable PyTorch compile cache for faster startup on subsequent runs
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_compile_cache
export TORCH_COMPILE_CACHE=1

# Use uv to run accelerate with proper dependencies
# --mixed_precision bf16 is now handled in the training script
uv run accelerate launch examples/wanvideo/model_training/train_inbetween_ti2v.py \
  --dataset_base_path /home/tjiao/cv_proj/dep/clipshots/videos/ClipShots_extracted/ClipShots/videos/train \
  --dataset_metadata_path /home/tjiao/cv_proj/dep/diffsynth/data/inbetween/train_transitions.csv \
  --dataset_repeat 1 \
  --height 480 \
  --width 832 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-InBetween_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --num_start_frames 32 \
  --num_end_frames 32 \
  --extra_inputs "input_image" \
  --save_steps 500 \
  --tensorboard_log_dir "./logs/tensorboard/wan22-ti2v-inbetween" \
  --dataset_num_workers 8 \
  --gradient_accumulation_steps 1 \
  --auto_resume

# ============================================================================
# Training Parameters Explanation:
# ============================================================================
#
# Model Configuration:
# - model_id_with_origin_paths: Specifies the Wan2.2-TI2V-5B model files to load
#   - models_t5_umt5-xxl-enc-bf16.pth: Text encoder
#   - diffusion_pytorch_model*.safetensors: DiT model weights
#   - Wan2.2_VAE.pth: Video VAE
#
# LoRA Configuration:
# - lora_base_model: Which model to apply LoRA to ("dit")
# - lora_target_modules: Modules to apply LoRA adapters
#   - q,k,v,o: Attention projections
#   - ffn.0,ffn.2: Feed-forward network layers
# - lora_rank: Rank of LoRA matrices (32 is a good balance)
#
# Logging & Checkpointing:
# - save_steps: Save checkpoint every N steps (default: 200)
# - tensorboard_log_dir: Directory for TensorBoard logs
# - auto_resume: Automatically resume from latest checkpoint if exists
# - To disable auto-resume, use --no_auto_resume
# - TensorBoard can be viewed with: tensorboard --logdir ./logs/tensorboard
#
# InBetween Task Configuration:
# - num_start_frames: Number of starting frames to use as condition (default: 8)
# - num_end_frames: Number of ending frames to use as condition (default: 8)
# - The model will learn to generate frames between the conditioned start and end
# - Latent frames = (num_frames - 1) / 4 + 1, so 8 raw frames ≈ 2-3 latent frames
#
# Data Configuration:
# - height/width: Training resolution (must be divisible by 16)
# - dataset_base_path: Root directory containing training videos
# - dataset_metadata_path: CSV file with video paths
#
# ============================================================================
# Metadata Format Examples:
# ============================================================================
#
# 1. Simple CSV (uses global num_frames=49, starts from frame 0):
#    video
#    videos/video1.mp4
#    videos/video2.mp4
#
# 2. CSV with per-video start_frame_index:
#    video,start_frame_index
#    videos/video1.mp4,0
#    videos/video2.mp4,100
#
# 3. CSV with per-video num_frames:
#    video,num_frames
#    videos/video1.mp4,49
#    videos/video2.mp4,81
#
# 4. CSV with all optional fields:
#    video,prompt,start_frame_index,num_frames
#    videos/video1.mp4,"A dog running",0,49
#    videos/video2.mp4,"",50,81
#
# ============================================================================
# Multi-GPU Training (Optional):
# ============================================================================
# For multi-GPU training, modify the script:
#
# #SBATCH --gpus=2
#
# uv run accelerate launch \
#     --multi_gpu \
#     --num_processes=2 \
#     --mixed_precision=bf16 \
#     examples/wanvideo/model_training/train_inbetween_ti2v.py \
#     ... (rest of arguments)
#
# ============================================================================
# Tips for Better Results:
# ============================================================================
#
# 1. Frame Count: Start with 8 start/end frames for stable conditioning
#    - More frames = better context but less "generation"
#    - Fewer frames = harder task but more creative freedom
#
# 2. Resolution: Start lower (480x832) then increase for fine-tuning
#    - 480x832: Faster training, good for initial experiments
#    - 720x1280: Higher quality, requires more VRAM
#
# 3. Learning Rate: 
#    - 1e-4 for LoRA training (default)
#    - 1e-5 to 5e-5 for full fine-tuning
#
# 4. LoRA Rank:
#    - 16-32: Faster training, less capacity
#    - 64-128: More capacity, may need lower learning rate
#
# 5. Checkpointing & Resume:
#    - Checkpoints are saved every 200 steps by default
#    - Training automatically resumes from latest checkpoint
#    - Checkpoints include: model weights, optimizer state, scheduler state
#    - To start fresh, delete or move the checkpoint-* folders
#
# ============================================================================
#
# ============================================================================
