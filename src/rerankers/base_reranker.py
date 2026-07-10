from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """
    Contract for anything that re-scores retrieved chunks against a query.

    Retrieval (embedding similarity) is fast but approximate — it can miss
    the most relevant chunk if its wording doesn't closely match the query.
    A reranker re-scores a small candidate set with a slower, more precise
    model, so the chunks that actually reach the LLM are the best ones,
    not just the first ones the embedding search happened to find.
    """

    @abstractmethod
    def rerank(self, query: str, chunks: list[str], top_k: int) -> list[str]:
        """Takes a query and candidate chunks, returns the top_k chunks reordered by relevance."""
        pass
