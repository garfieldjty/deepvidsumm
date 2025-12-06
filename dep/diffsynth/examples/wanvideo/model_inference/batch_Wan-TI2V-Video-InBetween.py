import csv
import os
import torch
from einops import rearrange
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d


def model_fn_wan_ti2v_inbetween(
    dit,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    image_latents: torch.Tensor = None,
    ti2v_start_latents: torch.Tensor = None,
    ti2v_end_latents: torch.Tensor = None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs
):
    """
    Model function for Wan2.2-TI2V-5B video in-between task.
    This function conditions on both starting and ending frames for video interpolation.
    """
    # Replace start portion of latents with clean conditioned latents
    if ti2v_start_latents is not None:
        num_start = ti2v_start_latents.shape[2]
        latents = latents.clone()
        latents[:, :, :num_start] = ti2v_start_latents
    
    # Replace end portion of latents with clean conditioned latents
    if ti2v_end_latents is not None:
        num_end = ti2v_end_latents.shape[2]
        latents = latents.clone()
        latents[:, :, -num_end:] = ti2v_end_latents
    
    # Timestep embedding
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Text embedding
    context = dit.text_embedding(context)
    
    x = latents
    
    # Patchify
    x = dit.patchify(x, image_latents)
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    # Position embeddings
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).contiguous().to(x.device, non_blocking=True)
    
    # Run transformer blocks
    for block in dit.blocks:
        x = block(x, context, t_mod, freqs)
    
    # Head and unpatchify
    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    
    output = -x
    output = output.to(latents.dtype)
    return output


# Paths
csv_path = "/workspace/deepvidsumm/dep/diffsynth/data/between/train_transitions.csv"
video_base_path = "/workspace/deepvidsumm/dep/clipshots/videos/ClipShots/videos/train/"
output_dir = "generated_inbetween_videos_wan"
os.makedirs(output_dir, exist_ok=True)

# Load the pipeline
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

# Optionally load LoRA weights
try:
    pipe.load_lora(
        pipe.dit,
        lora_config="./models/train/Wan2.2-TI2V-InBetween_lora/epoch-0.safetensors",
        alpha=1.0,
    )
except Exception as e:
    print(f"LoRA weights not loaded: {e}")

pipe.model_fn = model_fn_wan_ti2v_inbetween

# Read CSV and process first N samples after an initial offset
N = 1200  # Number of samples to process
SKIP = 500  # Number of rows to skip before processing
with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    processed = 0
    for idx, row in enumerate(reader):
        if idx < SKIP:
            continue  # Skip the initial rows
        if processed >= N:
            break
        if len(row) < 4:
            continue
        processed += 1
        video_name, start_frame, total_frames = row[0], int(row[2]), int(row[3])
        video_path = os.path.join(video_base_path, video_name)
        print(f"Processing {video_name} from frame {start_frame} to {start_frame + total_frames - 1}")
        try:
            input_video = VideoData(
                video_file=video_path,
                height=480,
                width=832,
            )
            # Extract frames for original cut
            all_frames = [input_video[i] for i in range(start_frame, start_frame + total_frames)]
            # Save original cut for comparison
            ori_out_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_ori.mp4")
            save_video(all_frames, ori_out_path, fps=15, quality=5)
            print(f"Saved original cut to {ori_out_path}")

            num_start_frames = min(40, total_frames // 3)
            num_end_frames = min(40, total_frames // 3)
            ti2v_start_video = all_frames[:num_start_frames]
            ti2v_end_video = all_frames[-num_end_frames:]
            print(f"Loaded {len(all_frames)} frames. Generating in-between frames...")
            video = pipe(
                prompt="",
                negative_prompt="Bright tones, overexposed, static, blurred details, worst quality, low quality",
                input_image=all_frames[0],  # First frame as TI2V input
                ti2v_start_video=ti2v_start_video,
                ti2v_end_video=ti2v_end_video,
                seed=0,
                tiled=True,
                num_frames=total_frames,
                cfg_scale=5,
                sigma_shift=8,
            )
            out_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_inbetween.mp4")
            save_video(video, out_path, fps=15, quality=5)
            print(f"Saved output to {out_path}")
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
print("Batch generation complete.")
