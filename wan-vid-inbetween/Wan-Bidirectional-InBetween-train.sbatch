#!/bin/bash
#SBATCH --job-name=bidir-inbetween   # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --gpus=2                     # number of GPUs per node (only valid under large/normal partition)
#SBATCH --time=24:00:00              # total run time limit (HH:MM:SS) - longer for 2 LoRAs
#SBATCH --partition=normal           # partition(large/normal/cpu) where you submit
#SBATCH --account=mscaisuperpod      # only required for multiple projects

# module purge                       # clear environment modules inherited from submission
# module load Anaconda3/2023.09-0    # load the exact modules required

# Print GPU info for debugging
echo "=============================================="
echo "BIDIRECTIONAL INBETWEENING TRAINING"
echo "=============================================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --list-gpus

# Launch multi-GPU training
uv run accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=bf16 \
    -m src.train_bidirectional_inbetween \
    --config config/bidirectional_inbetween_config.yaml
