import logging
import time
from pathlib import Path
from typing import Tuple

from app.analyzer.base import VisualAnalyzerBase
from app.composer.base import VideoComposerBase
from app.models import TimeRange, seconds_to_hhmmss
from app.preprocessor.base import PreprocessorBase
from app.pipeline.base import PipelineBase


logger = logging.getLogger(__name__)


class SimplePipeline(PipelineBase):
    """
    Orchestrates preprocessing -> analysis -> composition.
    """

    def __init__(
        self,
        preprocessor: PreprocessorBase,
        analyzer: VisualAnalyzerBase,
        composer: VideoComposerBase,
        workspace_root: Path = Path("runs"),
    ):
        self.preprocessor = preprocessor
        self.analyzer = analyzer
        self.composer = composer
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _session_dir(self) -> Path:
        return self.workspace_root / f"session_{int(time.time())}"

    def run(self, user_prompt: str, video_path: str, progress=None) -> Tuple[str, str]:
        def _progress(pct: float, desc: str = ""):
            if progress:
                progress(pct, desc=desc)

        session_dir = self._session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Starting pipeline session at %s", session_dir)

        _progress(0.1, "Extracting frames...")
        logger.info("Preprocessing video: %s", video_path)
        preprocess_result = self.preprocessor.preprocess(video_path, workspace_path=str(session_dir))

        duration = float(preprocess_result.video_info.get("duration", 0) or 0)
        _progress(0.45, "Analyzing frames for answer ranges...")
        logger.info("Analyzing frames; detected duration: %.2f seconds", duration)
        dump_path = session_dir / "analyzer_raw_response.json"
        req_dump_path = session_dir / "analyzer_request.json"
        ranges, note = self.analyzer.analyze(
            preprocess_result,
            user_prompt,
            response_dump_path=dump_path,
            request_dump_path=req_dump_path,
        )
        ranges = [r.clamp(duration) for r in ranges]
        if not ranges:
            ranges = [TimeRange(0.0, min(15.0, duration if duration > 0 else 15.0))]
            note += " (Defaulted to first 15 seconds because no ranges returned.)"
        logger.info("Selected %d ranges", len(ranges))

        _progress(0.7, "Cutting and stitching answer clip...")
        output_path = session_dir / "answer.mp4"
        logger.info("Composing output video to %s", output_path)
        self.composer.compose(video_path, ranges, output_path)

        readable_ranges = ", ".join(
            f"{seconds_to_hhmmss(r.start)}–{seconds_to_hhmmss(r.end)}" for r in ranges
        )
        status = (
            f"{note}\n"
            f"Selected ranges: {readable_ranges}\n"
            f"Output saved to: {output_path}"
        )

        _progress(1.0, "Done")
        return str(output_path), status
