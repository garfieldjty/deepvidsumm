# scripts/train_lora_inbetween.py

import argparse

from src.utils import load_config
from src.trainer import TrainConfig, InbetweenTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_inbetween_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Map YAML -> TrainConfig
    train_cfg = TrainConfig(
        base_model_path=cfg["base_model_path"],
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
        train_batch_size=cfg["train_batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        num_train_steps=cfg["num_train_steps"],
        learning_rate=cfg.get("learning_rate", 1e-4),
        seed=cfg.get("seed", 42),
        output_dir=cfg["output_dir"],
        lora_r=cfg.get("lora_r", 64),
        lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        transformer_precision=cfg.get("transformer_precision", "bf16"),
        vae_precision=cfg.get("vae_precision", "fp32"),
    )

    trainer = InbetweenTrainer(train_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
