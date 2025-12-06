from vbench import VBench

device = "cuda"  # or "cpu"
video_path = "/workspace/deepvidsumm/dep/diffsynth/generated_inbetween_videos_wan"
output_path = "/workspace/deepvidsumm/evaluation/vbench_results"
name = "evaluation_results"

vbench_instance = VBench(device, video_path, output_path)
vbench_instance.evaluate(
    videos_path = video_path,
    name = name,
    mode='custom_input',
    dimension_list = ['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'],
)
