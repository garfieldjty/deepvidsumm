import csv
import os
import torch
from tqdm import tqdm
from einops import rearrange
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d


def model_fn_wan_ti2v_inbetween(
    dit,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
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
    # Clone latents since we need to modify them
    if ti2v_start_latents is not None or ti2v_end_latents is not None:
        latents = latents.clone()
    
    # Replace start portion of latents with clean conditioned latents
    if ti2v_start_latents is not None:
        num_start = ti2v_start_latents.shape[2]
        latents[:, :, :num_start] = ti2v_start_latents
    
    # Replace end portion of latents with clean conditioned latents
    if ti2v_end_latents is not None:
        num_end = ti2v_end_latents.shape[2]
        latents[:, :, -num_end:] = ti2v_end_latents
    
    # Timestep embedding
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Text embedding
    context = dit.text_embedding(context)
    
    x = latents
    
    # Patchify - pass None for image_latents since we're not using TI2V image conditioning
    x = dit.patchify(x, None)
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


def compute_latent_frames(num_video_frames: int) -> int:
    """
    Compute number of latent frames from video frames.
    The VAE compresses temporally by factor of 4, with formula: (F - 1) // 4 + 1
    """
    return (num_video_frames - 1) // 4 + 1


def run_inbetween_inference(
    pipe,
    all_frames,
    num_start_frames,
    num_end_frames,
    prompt="",
    negative_prompt="",
    height=480,
    width=832,
    num_inference_steps=50,
    cfg_scale=5.0,
    sigma_shift=8.0,
    seed=0,
    tiled=True,
    tile_size=(30, 52),
    tile_stride=(15, 26),
):
    """
    Run video inbetweening inference with proper start/end frame conditioning.
    """
    num_frames = len(all_frames)
    
    # Set up scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=sigma_shift)
    
    # Check and resize dimensions
    height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
    
    # Get VAE parameters for proper noise generation
    z_dim = pipe.vae.model.z_dim  # Number of latent channels
    upsampling_factor = pipe.vae.upsampling_factor  # Spatial downsampling factor
    
    # Generate noise with correct dimensions
    latent_frames = compute_latent_frames(num_frames)
    noise_shape = (1, z_dim, latent_frames, height // upsampling_factor, width // upsampling_factor)
    noise = pipe.generate_noise(noise_shape, seed=seed, rand_device="cpu")
    
    # Encode prompt
    pipe.load_models_to_device(["text_encoder"])
    context_posi = pipe.prompter.encode_prompt(prompt, positive=True, device=pipe.device)
    if cfg_scale != 1.0:
        context_nega = pipe.prompter.encode_prompt(negative_prompt, positive=False, device=pipe.device)
    else:
        context_nega = None
    
    # Offload text encoder to save memory
    pipe.load_models_to_device([])
    torch.cuda.empty_cache()
    
    # Encode the full video and extract start/end latents
    pipe.load_models_to_device(["vae"])
    input_video = pipe.preprocess_video(all_frames)
    with torch.no_grad():
        full_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        full_latents = full_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # Offload VAE to save memory before loading DiT
    del input_video
    pipe.load_models_to_device([])
    torch.cuda.empty_cache()
    
    # Calculate latent frame counts
    num_start_latents = compute_latent_frames(num_start_frames)
    num_end_latents = compute_latent_frames(num_end_frames)
    
    # Extract start and end latents (CLEAN latents for conditioning)
    ti2v_start_latents = full_latents[:, :, :num_start_latents].clone()
    ti2v_end_latents = full_latents[:, :, -num_end_latents:].clone()
    
    print(f"Noise shape: {noise.shape}")
    print(f"Full latent shape: {full_latents.shape}")
    print(f"Start latents shape: {ti2v_start_latents.shape} ({num_start_latents} latent frames)")
    print(f"End latents shape: {ti2v_end_latents.shape} ({num_end_latents} latent frames)")
    
    # Verify shapes match
    assert noise.shape == full_latents.shape, f"Noise shape {noise.shape} != full_latents shape {full_latents.shape}"
    
    # Initialize latents with pure noise (not based on input video for middle frames)
    latents = noise.clone()
    
    # Load DiT for inference
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    
    # Denoising loop - use no_grad for inference to save memory
    progress_bar = tqdm(enumerate(pipe.scheduler.timesteps), total=len(pipe.scheduler.timesteps), desc="Denoising")
    with torch.no_grad():
        for progress_id, timestep in progress_bar:
            timestep_tensor = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            
            # Positive prediction with start/end conditioning
            noise_pred_posi = pipe.model_fn(
                dit=models["dit"],
                latents=latents,
                timestep=timestep_tensor,
                context=context_posi,
                ti2v_start_latents=ti2v_start_latents,
                ti2v_end_latents=ti2v_end_latents,
            )
            
            # Classifier-free guidance
            if cfg_scale != 1.0 and context_nega is not None:
                noise_pred_nega = pipe.model_fn(
                    dit=models["dit"],
                    latents=latents,
                    timestep=timestep_tensor,
                    context=context_nega,
                    ti2v_start_latents=ti2v_start_latents,
                    ti2v_end_latents=ti2v_end_latents,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            
            # Debug: compute what the training target would be at this timestep
            if progress_id % 10 == 0:
                # Training target is: noise - clean_latents (for flow matching)
                gt_target = noise - full_latents
                # Compare model output to GT target
                mse_to_gt = torch.nn.functional.mse_loss(noise_pred.float(), gt_target.float())
                # Also compute MSE of current latents to GT
                current_mse = torch.nn.functional.mse_loss(latents.float(), full_latents.float())
                print(f"\n[DEBUG] Step {progress_id}, timestep={timestep.item():.2f}")
                print(f"[DEBUG] noise_pred stats: mean={noise_pred.mean().item():.4f}, std={noise_pred.std().item():.4f}")
                print(f"[DEBUG] gt_target stats: mean={gt_target.mean().item():.4f}, std={gt_target.std().item():.4f}")
                print(f"[DEBUG] MSE to GT target: {mse_to_gt.item():.6f}")
                print(f"[DEBUG] Current latents MSE to GT: {current_mse.item():.6f}")
            
            # Scheduler step
            latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], latents)
            
            # Replace start/end latents with clean versions after each step
            # This ensures the conditioning frames remain unchanged
            latents[:, :, :num_start_latents] = ti2v_start_latents
            latents[:, :, -num_end_latents:] = ti2v_end_latents
    
    # Debug: Compare final latents to ground truth
    final_mse = torch.nn.functional.mse_loss(latents.float(), full_latents.float())
    middle_start = num_start_latents
    middle_end = latents.shape[2] - num_end_latents
    if middle_end > middle_start:
        middle_mse = torch.nn.functional.mse_loss(
            latents[:, :, middle_start:middle_end].float(), 
            full_latents[:, :, middle_start:middle_end].float()
        )
        print(f"\n[DEBUG] Final latents MSE to GT (full): {final_mse.item():.6f}")
        print(f"[DEBUG] Final latents MSE to GT (middle only): {middle_mse.item():.6f}")
    else:
        print(f"\n[DEBUG] Final latents MSE to GT (full): {final_mse.item():.6f}")
    
    # Decode latents to video
    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
    video = pipe.vae_output_to_video(video)
    pipe.load_models_to_device([])
    
    return video


# Paths
csv_path = "/workspace/deepvidsumm/dep/diffsynth/data/between/eval.csv"
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
lora_path = "./models/train/upload/model.safetensors"
if os.path.exists(lora_path):
    try:
        pipe.load_lora(
            pipe.dit,
            lora_config=lora_path,
            alpha=1.0,
        )
        print(f"LoRA weights loaded successfully from {lora_path}")
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
else:
    print(f"WARNING: LoRA file not found at {lora_path}. Running with base model only!")

# Set custom model function for inbetweening
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
        print(f"\n{'='*60}")
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

            # Calculate conditioning frame counts
            num_start_frames = 32
            num_end_frames = 32
            
            num_start_latents = compute_latent_frames(num_start_frames)
            num_end_latents = compute_latent_frames(num_end_frames)
            
            print(f"Loaded {len(all_frames)} frames.")
            print(f"Start conditioning: {num_start_frames} frames ({num_start_latents} latents)")
            print(f"End conditioning: {num_end_frames} frames ({num_end_latents} latents)")
            print(f"Generating in-between frames...")
            
            # Use our custom inference function that properly passes ti2v_start_latents and ti2v_end_latents
            video = run_inbetween_inference(
                pipe=pipe,
                all_frames=all_frames,
                num_start_frames=num_start_frames,
                num_end_frames=num_end_frames,
                prompt="",
                negative_prompt="Bright tones, overexposed, static, blurred details, worst quality, low quality",
                height=480,
                width=832,
                num_inference_steps=50,
                cfg_scale=1.0,
                sigma_shift=5.0,
                seed=0,
                tiled=True,
            )
            out_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_inbetween.mp4")
            save_video(video, out_path, fps=15, quality=5)
            print(f"Saved output to {out_path}")
        except Exception as e:
            import traceback
            print(f"Error processing {video_name}: {e}")
            traceback.print_exc()
        finally:
            # Clean up memory after each video
            torch.cuda.empty_cache()
            
print("\nBatch generation complete.")
