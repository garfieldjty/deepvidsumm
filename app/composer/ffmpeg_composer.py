import logging
import subprocess
from pathlib import Path
from typing import List

from app.models import TimeRange
from .base import VideoComposerBase


logger = logging.getLogger(__name__)


class FFMpegComposer(VideoComposerBase):
    """Cuts and stitches ranges using ffmpeg."""

    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        hwaccel: str | None = None,
        decoder: str | None = None,
        encoder: str | None = "libx264",
    ):
        self.ffmpeg_bin = ffmpeg_bin
        self.hwaccel = hwaccel
        self.decoder = decoder
        self.encoder = encoder

    def _run(self, cmd: List[str]):
        logger.info("Running command: %s", " ".join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
        return result

    def compose(self, video_path: str, ranges: List[TimeRange], output_path: Path) -> Path:
        workdir = output_path.parent
        workdir.mkdir(parents=True, exist_ok=True)

        segment_files = []
        for idx, rng in enumerate(ranges):
            if rng.end <= rng.start:
                # Ensure non-zero duration to avoid ffmpeg -to/-ss collision.
                rng = TimeRange(start=rng.start, end=rng.start + 0.5)
            duration = max(rng.end - rng.start, 0.5)

            seg_path = workdir / f"segment_{idx:02d}.mp4"
            cmd = [self.ffmpeg_bin, "-y"]
            if self.hwaccel:
                cmd.extend(["-hwaccel", self.hwaccel])
            if self.decoder:
                cmd.extend(["-c:v", self.decoder])
            cmd.extend(
                [
                    "-ss",
                    f"{rng.start:.3f}",
                    "-i",
                    video_path,
                    "-t",
                    f"{duration:.3f}",
                    "-probesize",
                    "5M",
                    "-analyzeduration",
                    "5M",
                    "-c:v",
                    self.encoder if self.encoder else "libx264",
                    "-an",
                    "-movflags",
                    "+faststart",
                    "-avoid_negative_ts",
                    "make_zero",
                    str(seg_path),
                ]
            )
            try:
                self._run(cmd)
            except Exception as exc:
                logger.warning("GPU-accelerated cut failed, retrying on CPU: %s", exc)
                fallback_cmd = [
                    self.ffmpeg_bin,
                    "-y",
                    "-ss",
                    f"{rng.start:.3f}",
                    "-i",
                    video_path,
                    "-t",
                    f"{duration:.3f}",
                    "-probesize",
                    "5M",
                    "-analyzeduration",
                    "5M",
                    "-c:v",
                    "libx264",
                    "-an",
                    "-movflags",
                    "+faststart",
                    "-avoid_negative_ts",
                    "make_zero",
                    str(seg_path),
                ]
                self._run(fallback_cmd)
            segment_files.append(seg_path.resolve())

        if not segment_files:
            raise RuntimeError("No segments were created to compose the final video.")

        concat_file = workdir / "concat.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        cmd = [self.ffmpeg_bin, "-y"]
        if self.hwaccel:
            cmd.extend(["-hwaccel", self.hwaccel])
        cmd.extend(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file.resolve()),
                "-c",
                "copy",
                str(output_path),
            ]
        )
        self._run(cmd)
        return output_path
