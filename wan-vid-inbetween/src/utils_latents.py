# src/utils_latents.py
import torch

def retrieve_latents(encoder_output, sample_mode="argmax"):
    """
    VAE encoder returns a distribution; we need latents of shape:
    [B, C, T_lat, H_lat, W_lat]
    """
    latents = encoder_output.latent_dist
    if sample_mode == "argmax":
        return latents.mode()
    else:
        return latents.sample()
