from abc import ABC, abstractmethod


class GUIBase(ABC):
    """Abstract class for user interfaces."""

    @abstractmethod
    def launch(self):
        """Start the UI server."""
        raise NotImplementedError
