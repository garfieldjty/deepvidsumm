import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict

from app.models import PreprocessResult, seconds_to_hhmmss
from .base import PreprocessorBase


logger = logging.getLogger(__name__)


class FramePreprocessor(PreprocessorBase):
    """Minimal ffmpeg/ffprobe-based frame extractor (audio is ignored)."""

    def __init__(
        self,
        workspace_path: str,
        target_fps: int = 2,
        target_width: int = 360,
        hwaccel: str | None = "auto",
        decoder: str | None = None,
        hw_output_format: str | None = None,
    ):
        self.workspace_path = workspace_path
        self.target_fps = target_fps
        self.target_width = target_width
        self.hwaccel = hwaccel
        self.decoder = decoder
        self.hw_output_format = hw_output_format
        os.makedirs(workspace_path, exist_ok=True)
        logger.info(
            "Preprocessor init: fps=%s width=%s hwaccel=%s decoder=%s hw_output_format=%s",
            target_fps,
            target_width,
            hwaccel,
            decoder,
            hw_output_format,
        )

    def _run(self, cmd):
        logger.info("Running command: %s", " ".join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
        return result.stdout

    def _probe(self, video_path: str) -> Dict[str, str]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,codec_name,pix_fmt,time_base",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            video_path,
        ]
        raw = self._run(cmd)
        data = json.loads(raw)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
        fps_str = stream.get("avg_frame_rate", "0/1")
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except Exception:
            fps = 0.0
        time_base = stream.get("time_base", "0/1")

        return {
            "width": str(stream.get("width", "")),
            "height": str(stream.get("height", "")),
            "fps": f"{fps:.3f}" if fps else "",
            "duration": fmt.get("duration", ""),
            "size": fmt.get("size", ""),
            "codec": stream.get("codec_name", ""),
            "pix_fmt": stream.get("pix_fmt", ""),
            "time_base": time_base,
        }

    def _extract_frames(self, video_path: str, frames_dir: Path) -> Dict[str, str]:
        frames_dir.mkdir(parents=True, exist_ok=True)
        pattern = frames_dir / "frame_%010d.jpg"
        if self.target_fps <= 0:
            raise ValueError("target_fps must be greater than zero.")

        # ffmpeg numbers frames with output PTS when -frame_pts is set; with the fps filter
        # applied, that time base is 1/target_fps.
        seconds_per_frame = 1.0 / float(self.target_fps)
        cmd_hw = ["ffmpeg", "-y"]
        if self.hwaccel:
            cmd_hw.extend(["-hwaccel", self.hwaccel])
        if self.hw_output_format:
            cmd_hw.extend(["-hwaccel_output_format", self.hw_output_format])
        if self.decoder:
            cmd_hw.extend(["-c:v", self.decoder])

        if self.hwaccel == "cuda":
            vf_hw = f"fps={self.target_fps},scale_cuda={self.target_width}:-2,hwdownload,format=rgb24"
        else:
            vf_hw = f"fps={self.target_fps},scale={self.target_width}:-2,format=rgb24"

        cmd_hw.extend(
            [
                "-i",
                video_path,
                "-frame_pts",
                "1",
                "-vf",
                vf_hw,
                "-probesize",
                "5M",
                "-analyzeduration",
                "5M",
                "-an",
                "-q:v",
                "2",
                str(pattern),
            ]
        )

        try:
            self._run(cmd_hw)
        except Exception as exc:
            logger.warning("Hardware-accelerated frame extraction failed, falling back to CPU: %s", exc)
            cmd_cpu = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-frame_pts",
                "1",
                "-vf",
                f"fps={self.target_fps},scale={self.target_width}:-2,format=rgb24",
                "-probesize",
                "5M",
                "-analyzeduration",
                "5M",
                "-an",
                "-q:v",
                "2",
                str(pattern),
            ]
            self._run(cmd_cpu)

        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        frame_map: Dict[str, str] = {}
        pts_pattern = re.compile(r"frame_(\d+)\.jpg$")
        for idx, frame_path in enumerate(frame_paths):
            match = pts_pattern.search(frame_path.name)
            if match:
                pts_val = int(match.group(1))
                timestamp_sec = pts_val * seconds_per_frame
            else:
                timestamp_sec = idx * seconds_per_frame

            ts_str = seconds_to_hhmmss(timestamp_sec)
            # Keep the timestamp in the filename for traceability.
            renamed = frame_path.with_name(f"frame_{idx:05d}_{ts_str}.jpg")
            try:
                frame_path.rename(renamed)
                final_path = renamed
            except Exception:
                # If rename fails, try copying to preserve the timestamp in the filename.
                try:
                    shutil.copy2(frame_path, renamed)
                    final_path = renamed
                except Exception:
                    final_path = frame_path

            frame_map[ts_str] = str(final_path)
        return frame_map

    def preprocess(self, video_path: str, workspace_path: str | None = None) -> PreprocessResult:
        workspace = Path(workspace_path or self.workspace_path)
        workspace.mkdir(parents=True, exist_ok=True)
        frames_dir = workspace / "frames"

        info = self._probe(video_path)

        frames = self._extract_frames(video_path, frames_dir)

        return PreprocessResult(
            video_info=info,
            frames=frames,
            frame_count=len(frames),
            frames_dir=str(frames_dir),
        )
