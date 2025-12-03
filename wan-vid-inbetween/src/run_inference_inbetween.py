# src/run_inference_inbetween.py
import argparse

from .inference import generate_inbetween_around_cut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--cut_frame_index", type=int, required=True)

    parser.add_argument("--start_frames", type=int, default=30)
    parser.add_argument("--mid_frames", type=int, default=60)
    parser.add_argument("--end_frames", type=int, default=30)

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--out_fps", type=int, default=24)
    parser.add_argument("--output_path", type=str, default="inbetween_output.mp4")

    parser.add_argument("--transformer_precision", type=str, default="bf16")
    parser.add_argument("--vae_precision", type=str, default="fp32")

    args = parser.parse_args()

    out = generate_inbetween_around_cut(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        video_path=args.video_path,
        cut_frame_index=args.cut_frame_index,
        start_frames=args.start_frames,
        mid_frames=args.mid_frames,
        end_frames=args.end_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        out_fps=args.out_fps,
        transformer_precision=args.transformer_precision,
        vae_precision=args.vae_precision,
        output_path=args.output_path,
    )
    print("Saved:", out)


if __name__ == "__main__":
    main()
