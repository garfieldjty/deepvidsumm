from dataclasses import dataclass, asdict
import re
from typing import Any, Dict, List


TIME_RE = re.compile(r"(?:(\d{2}):)?(\d{2}):(\d{2}(?:\.\d{1,3})?)")


@dataclass
class TimeRange:
    start: float
    end: float

    def clamp(self, duration: float) -> "TimeRange":
        """Clamp the range to a valid interval within the video duration."""
        start = max(0.0, min(self.start, duration))
        end = max(start, min(self.end, duration))
        return TimeRange(start=start, end=end)


def hhmmss_to_seconds(value: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    match = TIME_RE.match(value.strip())
    if not match:
        return 0.0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


@dataclass
class PreprocessResult:
    video_info: Dict[str, Any]
    frames: Dict[str, str]  # timestamp -> frame path
    frame_count: int
    frames_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessResult":
        return cls(**data)
