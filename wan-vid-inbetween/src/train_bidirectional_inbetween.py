# scripts/train_bidirectional_inbetween.py
"""
Training script for Bidirectional Video Inbetweening.

This trainer uses:
- Single LoRA for both forward and backward passes
- Pre-trained (frozen) Cumulative Softmax MLP for fusion weights

Training flow:
1. First train the fusion MLP using train_fusion_mlp.py
2. Then run this script with pretrained_fusion_mlp_path pointing to the MLP

Usage:
    # Single GPU
    python -m src.train_bidirectional_inbetween --config config/bidirectional_inbetween_config.yaml

    # Multi-GPU with accelerate
    accelerate launch -m src.train_bidirectional_inbetween --config config/bidirectional_inbetween_config.yaml
    
    # Resume from checkpoint
    python -m src.train_bidirectional_inbetween --config config/bidirectional_inbetween_config.yaml --resume latest
"""

import argparse

from src.utils import load_config
from src.trainer import BidirectionalTrainConfig, BidirectionalInbetweenTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train Bidirectional Video Inbetweening with Frozen Fusion MLP"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/bidirectional_inbetween_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint. Use 'latest' for most recent, or provide path.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # Override resume_from_checkpoint if provided via CLI
    if args.resume:
        cfg["resume_from_checkpoint"] = args.resume

    # Map YAML -> BidirectionalTrainConfig
    train_cfg = BidirectionalTrainConfig(
        # Model
        base_model_path=cfg["base_model_path"],
        transformer_precision=cfg.get("transformer_precision", "bf16"),
        vae_precision=cfg.get("vae_precision", "fp32"),
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        
        # Data
        data_root=cfg["data_root"],
        video_glob=cfg.get("video_glob", "**/*.mp4"),
        clip_num_frames=cfg["clip_num_frames"],
        height=cfg["resolution"]["height"],
        width=cfg["resolution"]["width"],
        start_frames=cfg["start_frames"],
        mid_frames=cfg["mid_frames"],
        end_frames=cfg["end_frames"],
        cut_annotations_path=cfg.get("cut_annotations_path", None),
        use_cut_focused_sampling=cfg.get("use_cut_focused_sampling", True),
        cut_focus_prob=cfg.get("cut_focus_prob", 0.7),
        
        # Training
        train_batch_size=cfg["train_batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        num_train_steps=cfg["num_train_steps"],
        learning_rate=cfg.get("learning_rate", 1e-4),
        seed=cfg.get("seed", 42),
        output_dir=cfg["output_dir"],
        
        # LoRA
        lora_r=cfg.get("lora_r", 64),
        lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        
        # Fusion MLP (pre-trained and frozen)
        fusion_hidden_dim=cfg.get("fusion_hidden_dim", 256),
        fusion_num_layers=cfg.get("fusion_num_layers", 3),
        cnn_feature_dim=cfg.get("cnn_feature_dim", 64),
        pretrained_fusion_mlp_path=cfg.get("pretrained_fusion_mlp_path", None),
        
        # Loss weights
        loss_weight_fwd=cfg.get("loss_weight_fwd", 1.0),
        loss_weight_bwd=cfg.get("loss_weight_bwd", 1.0),
        
        # Sparse generation threshold
        weight_threshold=cfg.get("weight_threshold", 0.3),
        
        # Logging
        log_dir=cfg.get("log_dir", "./logs"),
        log_every_n_steps=cfg.get("log_every_n_steps", 10),
        
        # Checkpointing
        save_every_n_steps=cfg.get("save_every_n_steps", 1000),
        resume_from_checkpoint=cfg.get("resume_from_checkpoint", None),
    )

    # Validate that pretrained MLP path is provided
    if not train_cfg.pretrained_fusion_mlp_path:
        print("\n" + "=" * 60)
        print("WARNING: No pretrained_fusion_mlp_path specified!")
        print("The fusion MLP will be randomly initialized.")
        print("For best results, first train the MLP using:")
        print("  python -m src.train_fusion_mlp --config config/fusion_mlp_config.yaml")
        print("=" * 60 + "\n")

    print("\n" + "=" * 60)
    print("BIDIRECTIONAL INBETWEENING TRAINING")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Output: {train_cfg.output_dir}")
    print(f"LoRA rank: {train_cfg.lora_r}")
    print(f"Fusion MLP: {train_cfg.fusion_num_layers} layers, {train_cfg.fusion_hidden_dim} hidden dim")
    if train_cfg.pretrained_fusion_mlp_path:
        print(f"Fusion MLP weights: {train_cfg.pretrained_fusion_mlp_path} (frozen)")
    else:
        print(f"Fusion MLP weights: random init (WARNING: not recommended)")
    print(f"Weight threshold: {train_cfg.weight_threshold}")
    print(f"Loss weights: fwd={train_cfg.loss_weight_fwd}, bwd={train_cfg.loss_weight_bwd}")
    print("=" * 60 + "\n")

    trainer = BidirectionalInbetweenTrainer(train_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
