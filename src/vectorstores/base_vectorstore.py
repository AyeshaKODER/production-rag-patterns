from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """
    Contract for anything that stores embeddings and lets you search them.
    Chroma, Pinecone, FAISS would all implement this same interface.

    metadatas is optional so existing callers that don't pass it keep
    working — only code that wants source tracking (e.g. "which file did
    this chunk come from") needs to use it.
    """

    @abstractmethod
    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict] = None) -> None:
        """Store texts, their embeddings, and optional per-chunk metadata."""
        pass

    @abstractmethod
    def query(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """
        Given a query vector, return the top_k most similar items as
        dicts: {"text": str, "metadata": dict}. metadata is {} if none
        was stored for that chunk.
        """
        pass