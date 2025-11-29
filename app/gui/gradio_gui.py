from pathlib import Path
from typing import Tuple

import gradio as gr

from app.gui.base import GUIBase
from app.pipeline.base import PipelineBase


class GradioGUI(GUIBase):
    """Gradio implementation wiring together preprocessing, analysis, and composition."""

    def __init__(
        self,
        pipeline: PipelineBase,
    ):
        self.pipeline = pipeline

    def _process_video(self, user_prompt: str, video_file, progress=gr.Progress(track_tqdm=True)) -> Tuple[str | None, str]:
        if not user_prompt:
            return None, "Please enter a prompt."
        if not video_file:
            return None, "Please upload a video."

        try:
            video_path = getattr(video_file, "name", None) or str(video_file)
            output_path, status = self.pipeline.run(user_prompt, video_path, progress=progress)
            return output_path, status
        except Exception as exc:  # noqa: BLE001
            return None, f"Error while processing: {exc}"

    def launch(self):
        with gr.Blocks(title="deepvidsumm") as demo:
            gr.Markdown(
                "## deepvidsumm\n"
                "Upload a video, enter a prompt, and get a short answer-focused clip.\n"
                "If an OpenRouter API key is available, the vision model will be used; otherwise a mock heuristic runs."
            )
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", placeholder="e.g., When does Mr. Sloth speak?", lines=2)
            run_btn = gr.Button("Generate short clip", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Input video**")
                    video = gr.Video(label="Upload video", sources=["upload"], format="mp4")
                with gr.Column():
                    gr.Markdown("**Answer video**")
                    output_video = gr.Video(label="Answer clip")
            status = gr.Textbox(label="Status", interactive=False)

            run_btn.click(fn=self._process_video, inputs=[prompt, video], outputs=[output_video, status])

        import os

        demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
