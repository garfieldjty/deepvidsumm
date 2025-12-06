import logging
from pathlib import Path

from app.analyzer.openrouter_analyzer import OpenRouterVideoAnalyzer
from app.composer.ffmpeg_composer import FFMpegComposer
from app.gui.gradio_gui import GradioGUI
from app.pipeline.vid_clip_pipeline import VideoClipPipeline
from app.preprocessor.frame_preprocessor import FramePreprocessor


def analyzer_factory():
    logging.info("Initializing OpenRouterVideoAnalyzer")
    return OpenRouterVideoAnalyzer()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Use a stable workspace for session assets; UI will create per-run subdirs.
    workspace = Path("runs") / "latest"
    logging.info("Workspace root: %s", workspace)

    preprocessor = FramePreprocessor(workspace_path=str(workspace))
    composer = FFMpegComposer()
    analyzer = analyzer_factory()
    pipeline = VideoClipPipeline(
        preprocessor=preprocessor,
        analyzer=analyzer,
        composer=composer,
        workspace_root=Path("runs"),
    )
    gui = GradioGUI(pipeline=pipeline)
    gui.launch()


if __name__ == "__main__":
    main()
