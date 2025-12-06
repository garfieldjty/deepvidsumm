from abc import ABC, abstractmethod
from typing import Tuple


class PipelineBase(ABC):
    """Abstract pipeline interface wiring preprocessing, analysis, and composition."""

    @abstractmethod
    def run(self, user_prompt: str, video_path: str, progress=None) -> Tuple[str, str]:
        """Execute the full pipeline and return (output_video_path, status_text)."""
        raise NotImplementedError
