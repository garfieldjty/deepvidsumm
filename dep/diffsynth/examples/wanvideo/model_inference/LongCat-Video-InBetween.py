import torch
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, model_fn_longcat_video_inbetween


# Load the pipeline
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="meituan-longcat/LongCat-Video", origin_file_pattern="dit/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

# Load the trained LoRA weights (optional - comment out if using base model only)
pipe.load_lora(
    pipe.dit,
    lora_config="./models/train/LongCat-Video-InBetween_lora/epoch-4.safetensors",
    alpha=1.0,
)

# Override model function to use in-between mode
pipe.model_fn = model_fn_longcat_video_inbetween

# Load input video
print("Loading input video...")
input_video = VideoData(
    video_file="/workspace/deepvidsumm/dep/diffsynth/data/between/4001498009.mp4",
    height=480,
    width=832,
)
    
# Extract start and end frames (121 frames each, must satisfy (num_frames-1) % 4 == 0)
num_start_frames = 171
num_end_frames = 171
total_frames = 361


# Get frames from the video (starting from frame 38)
all_frames = [input_video[i] for i in range(38, 38 + total_frames)]
longcat_start_video = all_frames[:num_start_frames]
longcat_end_video = all_frames[-num_end_frames:]

print(f"Loaded {len(all_frames)} frames from video")
print(f"Start frames: {num_start_frames}, End frames: {num_end_frames}")
print(f"In-between frames to generate: {total_frames - num_start_frames - num_end_frames}")

# Generate in-between video
print("Generating in-between frames...")
video = pipe(
    prompt="",  # Add your prompt here if needed
    negative_prompt="Bright tones, overexposed, static, blurred details, worst quality, low quality",
    longcat_start_video=longcat_start_video,
    longcat_end_video=longcat_end_video,
    seed=0,
    tiled=True,
    num_frames=total_frames,
    cfg_scale=2,
    sigma_shift=1,
)

# Save the output
print("Saving video...")
save_video(video, "inbetween_output.mp4", fps=15, quality=5)
print("Done! Output saved to inbetween_output.mp4")
