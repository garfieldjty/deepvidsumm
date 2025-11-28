#!/bin/bash
#SBATCH --job-name=test          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=1                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal  # partition(large/normal/cpu) where you submit
#SBATCH --account=mscaisuperpod      # only require for multiple projects

# module purge                     # clear environment modules inherited from submission
# module load Anaconda3/2023.09-0  # load the exact modules required

# python test.py

# Use uv to run accelerate with proper dependencies
uv run accelerate launch examples/wanvideo/model_training/train_inbetween.py \
  --dataset_base_path /home/tjiao/cv_proj/dep/clipshots/videos/ClipShots_extracted/ClipShots/videos/train \
  --dataset_metadata_path /home/tjiao/cv_proj/dep/diffsynth/data/inbetween/metadata.csv \
  --dataset_repeat 2 \
  --model_id_with_origin_paths "meituan-longcat/LongCat-Video:dit/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LongCat-Video-InBetween_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "adaLN_modulation.1,attn.qkv,attn.proj,cross_attn.q_linear,cross_attn.kv_linear,cross_attn.proj,ffn.w1,ffn.w2,ffn.w3" \
  --lora_rank 32 \
  --num_start_frames 120 \
  --num_end_frames 120

# Notes:
# - num_start_frames: Number of starting frames to use as condition (default: 1)
# - num_end_frames: Number of ending frames to use as condition (default: 1)
# - The model will learn to generate the frames in between the conditioned start and end frames
# - For better results, you can increase num_start_frames and num_end_frames (e.g., 3-5 frames each)
# - Adjust num_frames to match your dataset (total frames including start and end)
# - The actual frames to be generated will be: num_frames - num_start_frames - num_end_frames
#
# Metadata format examples:
# 1. Simple CSV (uses global num_frames=49, starts from frame 0):
#    video
#    videos/video1.mp4
#    videos/video2.mp4
#
# 2. CSV with per-video start_frame_index (optional):
#    video,start_frame_index
#    videos/video1.mp4,0
#    videos/video2.mp4,100
#
# 3. CSV with per-video num_frames (optional):
#    video,num_frames
#    videos/video1.mp4,49
#    videos/video2.mp4,81
#
# 4. CSV with all optional fields:
#    video,prompt,start_frame_index,num_frames
#    videos/video1.mp4,"A dog running",0,49
#    videos/video2.mp4,"",50,81
#    videos/video3.mp4,"A cat",100,65
