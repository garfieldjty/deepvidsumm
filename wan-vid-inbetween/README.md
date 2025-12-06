# Wan Video Inbetweening

A video inbetweening framework based on the Wan2.2 model. Given start and end video segments, the model generates intermediate frames to create smooth transitions between them.

## Features

- **LoRA-based fine-tuning** of Wan2.2-TI2V-5B for video inbetweening
- **Unidirectional and Bidirectional** training modes
- **Cumulative Softmax Fusion Network** for monotonic temporal fusion (bidirectional mode)
- **Cut-focused sampling** using ClipShots annotations
- **Multi-GPU training** support via Accelerate

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA-capable GPU (24GB+ VRAM recommended)

## Project Structure

```
wan-vid-inbetween/
├── README.md
├── pyproject.toml
├── config/
│   ├── default_inbetween_config.yaml      # Standard unidirectional config
│   ├── bidirectional_inbetween_config.yaml # Bidirectional config
│   ├── fast_inbetween_config.yaml          # Faster training settings
│   └── fusion_net_config.yaml              # Fusion network training config
├── annotations/
│   ├── train_videos.json
│   ├── test_videos.json
│   └── all_videos.json
├── src/
│   ├── dataset.py                  # Video dataset and data loading
│   ├── models.py                   # Model definitions
│   ├── trainer.py                  # Training loop implementations
│   ├── inference.py                # Inference utilities
│   ├── evaluate_inbetween.py       # Evaluation metrics
│   ├── train_lora_inbetween.py     # Unidirectional LoRA training script
│   ├── train_bidirectional_inbetween.py  # Bidirectional training script
│   ├── train_fusion_net.py         # Fusion network pre-training script
│   ├── run_inference_inbetween.py  # Inference script
│   ├── wan_condition_transformer.py # Modified Wan transformer
│   ├── utils.py                    # General utilities
│   └── utils_latents.py            # Latent space utilities
├── scripts/
│   └── generate_video_annotations.py
├── run_inference.sh                # Inference runner script
├── run_evaluation.sh               # Evaluation runner script
├── Wan-Video-InBetween-train.sbatch        # SLURM job for unidirectional
└── Wan-Bidirectional-InBetween-train.sbatch # SLURM job for bidirectional
```

## Training

### 1. Unidirectional LoRA Training

Standard forward-only inbetweening:

```bash
# Single GPU
python -m src.train_lora_inbetween --config config/default_inbetween_config.yaml

# Multi-GPU with accelerate
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 \
    -m src.train_lora_inbetween --config config/default_inbetween_config.yaml

# Resume from checkpoint
python -m src.train_lora_inbetween --config config/default_inbetween_config.yaml --resume latest
```

### 2. Bidirectional Training (Two-Stage)

Bidirectional training uses both forward and backward temporal context for better results.

**Stage 1: Train the Fusion Network**
```bash
python -m src.train_fusion_net --config config/fusion_net_config.yaml
```

**Stage 2: Train Bidirectional LoRA with Frozen Fusion Network**
```bash
# Single GPU
python -m src.train_bidirectional_inbetween --config config/bidirectional_inbetween_config.yaml

# Multi-GPU
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 \
    -m src.train_bidirectional_inbetween --config config/bidirectional_inbetween_config.yaml
```

### SLURM Cluster

```bash
# Unidirectional
sbatch Wan-Video-InBetween-train.sbatch

# Bidirectional
sbatch Wan-Bidirectional-InBetween-train.sbatch
```

## Inference

### Using the inference script

```bash
# Edit run_inference.sh with your paths, then:
./run_inference.sh
```

### Python API

```python
from src.inference import load_wan_with_lora, load_wan_bidirectional

# Unidirectional model
vae, transformer, scheduler = load_wan_with_lora(
    base_model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    lora_path="outputs/inbetween_lora/",
    transformer_precision="bf16",
)

# Bidirectional model
vae, transformer, scheduler, fusion_net, _, _ = load_wan_bidirectional(
    base_model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    lora_fwd_path="outputs/bidirectional/lora_fwd",
    lora_bwd_path="outputs/bidirectional/lora_bwd",
    fusion_net_path="outputs/bidirectional/fusion_net.pt",
)
```

## Configuration

Key configuration options in YAML files:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_model_path` | Hugging Face model ID or local path | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` |
| `transformer_precision` | Transformer dtype (`bf16`, `fp16`, `no`) | `bf16` |
| `vae_precision` | VAE dtype (`fp32`, `fp16`) | `fp32` |
| `attn_implementation` | Attention type (`flash_attention_2`, `sdpa`, `eager`) | `sdpa` |
| `clip_num_frames` | Total frames per sample | `120` |
| `start_frames` | Conditioning frames from start | `30` |
| `mid_frames` | Frames to generate | `60` |
| `end_frames` | Conditioning frames from end | `30` |
| `lora_r` | LoRA rank | `128` |
| `lora_alpha` | LoRA alpha | `256` |
| `use_cut_focused_sampling` | Sample around annotated cuts | `true` |

## Evaluation

```bash
./run_evaluation.sh
```

The evaluation script computes metrics on test videos with known cuts and compares generated inbetween frames against ground truth.
