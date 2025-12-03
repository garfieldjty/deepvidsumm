import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.trainer import TrainConfig, InbetweenTrainer

# Create a minimal config for testing
cfg = TrainConfig(
    base_model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    data_root="/root/deepvidsumm/dep/clipshots/videos/ClipShots/videos/train",
    video_glob="**/*.mp4",
    clip_num_frames=120,
    height=384,
    width=672,
    start_frames=30,
    mid_frames=60,
    end_frames=30,
    cut_annotations_path="./data/train_middle_frames.json",
    use_cut_focused_sampling=True,
    cut_focus_prob=0.7,
    train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_steps=10,  # Just a few steps for testing
    learning_rate=1e-4,
    seed=42,
    output_dir="./test_output",
    lora_r=64,
    lora_alpha=128,
    lora_dropout=0.0,
    transformer_precision="bf16",
    vae_precision="fp32",
)

print("Testing GPU usage...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

# Try to create trainer
try:
    trainer = InbetweenTrainer(cfg)
    print("Trainer created successfully")
    
    # Check device placement
    print(f"\nDevice checks:")
    print(f"Transformer device: {next(trainer.transformer.parameters()).device}")
    print(f"VAE device: {next(trainer.vae.parameters()).device}")
    print(f"Accelerator device: {trainer.acc.device}")
    
    # Try to run one training step
    print("\nRunning one training step...")
    trainer.train()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()