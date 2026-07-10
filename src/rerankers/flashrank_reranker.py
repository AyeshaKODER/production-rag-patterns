from flashrank import Ranker, RerankRequest
from src.rerankers.base_reranker import BaseReranker


class FlashRankReranker(BaseReranker):
    """
    FlashRank-backed reranker. Chosen per the enterprise RAG stack
    referenced by Krish Naik's course: no Torch/Transformers dependency,
    runs on CPU, ~4MB default model — a good fit for this project's scale.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, chunks: list[str], top_k: int) -> list[str]:
        passages = [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]
        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)
        # results are sorted by relevance score, descending
        return [r["text"] for r in results[:top_k]]
