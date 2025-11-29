from abc import ABC, abstractmethod
from typing import List, Tuple
from app.models import TimeRange


class VisualAnalyzerBase(ABC):
    """Abstract class for producing answer time ranges from video artifacts."""

    @abstractmethod
    def analyze(self, preprocess_result, user_prompt: str) -> Tuple[List[TimeRange], str]:
        """Return ranges and a status note."""
        raise NotImplementedError
