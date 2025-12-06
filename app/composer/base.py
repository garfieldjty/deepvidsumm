from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from app.models import TimeRange


class VideoComposerBase(ABC):
    """Abstract class for assembling the final answer clip."""

    @abstractmethod
    def compose(self, video_path: str, ranges: List[TimeRange], output_path: Path) -> Path:
        """Produce an output video containing the requested ranges."""
        raise NotImplementedError
