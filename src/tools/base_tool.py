from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    Contract for anything an agent can call to take action in the world.
    Same pattern as BaseLLM/BaseEmbedder — the agent's reasoning loop
    depends only on this interface, so new tools can be added without
    touching the agent's core logic.
    """

    name: str
    description: str

    @abstractmethod
    def run(self, input_str: str) -> str:
        """Executes the tool with a plain string input, returns a plain string result."""
        pass