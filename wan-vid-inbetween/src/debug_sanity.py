# src/debug_sanity.py
"""
Quick sanity checks for the custom Wan condition transformer.

Run from repo root as:

    python -m src.debug_sanity

Checks:
- Shapes are consistent for [start, mid, end] layout.
- conditioning_mask is accepted and broadcast to token level.
- Changing mid inputs does NOT change the cond outputs.
- Changing mid inputs DOES change the mid outputs.
"""

import torch

from src.wan_condition_transformer import WanTransformer3DModel


def build_tiny_transformer():
    """
    Build a small WanTransformer3DModel just to test the conditioning behavior.
    This does NOT load pretrained weights; it's random, purely for shape / logic.
    """
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),        # temporal patch = 1 -> T_tokens == T_frames
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=8,
        out_channels=8,
        text_dim=32,
        freq_dim=32,
        ffn_dim=128,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        image_dim=None,
        added_kv_proj_dim=None,
        rope_max_seq_len=256,
        pos_embed_seq_len=None,
    )
    return model


@torch.no_grad()
def sanity_check_mask_behavior():
    """
    Core behavioral test:

    - We create two inputs:
        * inp1: [start, mid, end]
        * inp2: same as inp1, but with a DIFFERENT mid segment

    - With conditioning_mask marking start+end as conditioning:
        * outputs for start+end MUST be identical between runs
        * outputs for mid SHOULD differ between runs

    If this holds, cond tokens are not influenced by mid tokens,
    but mid tokens can still use cond tokens as context.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small toy sizes
    B, C, T, H, W = 2, 8, 10, 16, 16
    # Split into start / mid / end along time
    s, m, e = 3, 4, 3
    assert s + m + e == T

    model = build_tiny_transformer().to(device)
    model.eval()

    # Hidden states with true temporal order [start, mid, end]
    base = torch.randn(B, C, T, H, W, device=device)

    # Two variants that only differ in the mid region
    inp1 = base.clone()
    inp2 = base.clone()
    inp2[:, :, s:s + m] = torch.randn_like(inp2[:, :, s:s + m])

    # conditioning_mask: True for start+end, False for mid
    conditioning_mask = torch.zeros(B, T, device=device, dtype=torch.bool)
    conditioning_mask[:, :s] = True
    conditioning_mask[:, s + m:] = True

    # Dummy timesteps and encoder_hidden_states
    t = torch.randint(0, 1000, (B,), device=device)
    enc_states = torch.zeros(B, 1, model.config.text_dim, device=device, dtype=model.dtype)

    out1 = model(
        hidden_states=inp1,
        timestep=t,
        encoder_hidden_states=enc_states,
        conditioning_mask=conditioning_mask,
        return_dict=True,
    ).sample  # [B, C, T, H, W]

    out2 = model(
        hidden_states=inp2,
        timestep=t,
        encoder_hidden_states=enc_states,
        conditioning_mask=conditioning_mask,
        return_dict=True,
    ).sample  # [B, C, T, H, W]

    # Split outputs
    out1_start, out1_mid, out1_end = out1[:, :, :s], out1[:, :, s:s + m], out1[:, :, s + m:]
    out2_start, out2_mid, out2_end = out2[:, :, :s], out2[:, :, s:s + m], out2[:, :, s + m:]

    # 1) Conditioning outputs should be identical (self-att only)
    start_close = torch.allclose(out1_start, out2_start, atol=1e-5, rtol=1e-5)
    end_close = torch.allclose(out1_end, out2_end, atol=1e-5, rtol=1e-5)

    # 2) Mid outputs should differ (depend on mid inputs)
    mid_equal = torch.allclose(out1_mid, out2_mid, atol=1e-5, rtol=1e-5)

    print("=== Sanity Check: Mask Behavior ===")
    print(f"start outputs identical? {start_close}")
    print(f"end   outputs identical? {end_close}")
    print(f"mid   outputs identical? {mid_equal} (expected: False)")
    print()

    assert start_close, "Start (conditioning) outputs changed when mid inputs changed."
    assert end_close, "End (conditioning) outputs changed when mid inputs changed."
    assert not mid_equal, "Mid outputs did NOT change when mid inputs changed; mask behavior is wrong."

    print("Mask behavior looks correct ✅")


@torch.no_grad()
def sanity_check_shapes():
    """
    Simple shape check that mirrors the training-time logic:

    - build [start, mid, end] latents
    - build conditioning_mask
    - run transformer and verify output shape
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, T, H, W = 1, 8, 12, 16, 16
    s, m, e = 3, 6, 3
    assert s + m + e == T

    model = build_tiny_transformer().to(device)
    model.eval()

    latents = torch.randn(B, C, T, H, W, device=device)

    # conditioning_mask: start+end True, mid False
    conditioning_mask = torch.zeros(B, T, device=device, dtype=torch.bool)
    conditioning_mask[:, :s] = True
    conditioning_mask[:, s + m:] = True

    t = torch.randint(0, 1000, (B,), device=device)
    enc_states = torch.zeros(B, 1, model.config.text_dim, device=device, dtype=model.dtype)

    out = model(
        hidden_states=latents,
        timestep=t,
        encoder_hidden_states=enc_states,
        conditioning_mask=conditioning_mask,
        return_dict=True,
    ).sample

    print("=== Sanity Check: Shapes ===")
    print(f"in  shape: {latents.shape}")
    print(f"out shape: {out.shape}")
    assert out.shape == latents.shape, "Output shape does not match input shape."
    print("Shape check passed ✅\n")


def main():
    sanity_check_shapes()
    sanity_check_mask_behavior()


if __name__ == "__main__":
    main()
