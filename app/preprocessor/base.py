from abc import ABC, abstractmethod
from typing import Any

from app.models import PreprocessResult

class PreprocessorBase(ABC):
    """Abstract preprocessor that prepares videos for downstream analysis."""

    @abstractmethod
    def preprocess(self, video_path: str, workspace_path: str | None = None) -> PreprocessResult:
        """Extract frames and return a structured result."""
        raise NotImplementedError
