from openai import OpenAI
from .base_embedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI-backed embedder. Not used in the active pipeline (we're on
    LocalEmbedder for cost reasons), kept here to demonstrate that
    BaseEmbedder can be swapped to a hosted provider without touching
    pipeline.py — only this concrete class changes.
    """

    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None):
        self.client = OpenAI(api_key=api_key)  # falls back to OPENAI_API_KEY env var if None
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding