from typing import Tuple, Optional
import torch

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
)
from peft import LoraConfig, get_peft_model

try:
    # Newer PEFT: has TaskType enum
    from peft import TaskType
    _DEFAULT_TASK_TYPE = TaskType.FEATURE_EXTRACTION
except Exception:
    # Older PEFT: use plain string; must match allowed list in your error
    _DEFAULT_TASK_TYPE = "FEATURE_EXTRACTION"

from .wan_condition_transformer import WanTransformer3DModel


def load_wan_components(
    base_model_path: str,
    transformer_precision: str = "bf16",
    vae_precision: str = "fp32",
):
    dtype_t = torch.bfloat16 if transformer_precision == "bf16" else torch.float16
    dtype_vae = torch.float32 if vae_precision == "fp32" else torch.float16

    vae = AutoencoderKLWan.from_pretrained(
        base_model_path, subfolder="vae", torch_dtype=dtype_vae
    )
    transformer = WanTransformer3DModel.from_pretrained(
        base_model_path, subfolder="transformer", torch_dtype=dtype_t
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model_path, subfolder="scheduler"
    )

    # No text encoder needed for unconditional inbetweening
    return vae, transformer, scheduler


def add_lora_to_transformer(transformer, r, alpha, dropout):
    """
    Attach LoRA adapters to Wan transformer:

    - Attention projections: to_q / to_k / to_v
    - FeedForward MLP linears inside `ffn`:
        * ffn.net.0.proj  (GELU block's linear)
        * ffn.net.2       (output linear)

    PEFT matches target modules by substring on module names, so these
    strings are chosen to be specific enough for the Wan blocks.
    """
    target_modules = [
        # self-attention / cross-attention
        "to_q",
        "to_k",
        "to_v",
        # feed-forward inner linears
        "ffn.net.0.proj",
        "ffn.net.2",
    ]

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )
    return get_peft_model(transformer, config)
