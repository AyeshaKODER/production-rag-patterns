from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Contract for anything that generates text from a prompt.

    Any LLM (Groq, OpenAI, local model, etc.) must implement generate(),
    or Python will refuse to let you instantiate it. This is what lets
    pipeline.py call self.llm.generate(...) without ever knowing or
    caring which provider is actually behind it.
    """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Takes a prompt string, returns the generated text response."""
        pass