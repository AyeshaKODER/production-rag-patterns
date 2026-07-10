from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Contract for anything that turns text into a vector embedding.
    Any embedder (OpenAI, local model, etc.) must implement embed(),
    or Python will refuse to let you create an object from it.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Takes a string, returns a vector representing its meaning."""
        pass
