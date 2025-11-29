import csv
import os
import torch
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, model_fn_longcat_video_inbetween

# Paths
csv_path = "/workspace/deepvidsumm/dep/diffsynth/data/between/train_transitions.csv"
video_base_path = "/workspace/deepvidsumm/dep/clipshots/videos/ClipShots/videos/train/"
output_dir = "generated_inbetween_videos"
os.makedirs(output_dir, exist_ok=True)

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

# Optionally load LoRA weights
try:
    pipe.load_lora(
        pipe.dit,
        lora_config="./models/train/LongCat-Video-InBetween_lora/epoch-0.safetensors",
        alpha=1.0,
    )
except Exception as e:
    print(f"LoRA weights not loaded: {e}")

pipe.model_fn = model_fn_longcat_video_inbetween

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
            longcat_start_video = all_frames[:num_start_frames]
            longcat_end_video = all_frames[-num_end_frames:]
            print(f"Loaded {len(all_frames)} frames. Generating in-between frames...")
            video = pipe(
                prompt="",
                negative_prompt="Bright tones, overexposed, static, blurred details, worst quality, low quality",
                longcat_start_video=longcat_start_video,
                longcat_end_video=longcat_end_video,
                seed=0,
                tiled=True,
                num_frames=total_frames,
                cfg_scale=2,
                sigma_shift=1,
            )
            out_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_inbetween.mp4")
            save_video(video, out_path, fps=15, quality=5)
            print(f"Saved output to {out_path}")
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
print("Batch generation complete.")
