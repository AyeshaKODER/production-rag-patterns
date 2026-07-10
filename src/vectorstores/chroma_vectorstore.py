import chromadb
from openai import embeddings
from src.vectorstores.base_vectorstore import BaseVectorStore
import uuid

class ChromaVectorStore(BaseVectorStore):
    """Concrete vector store using Chroma (local, in-memory)."""

    def __init__(self, collection_name: str = "rag_collection"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, texts: list[str], embeddings: list[list[float]]) -> None:
        ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(documents=texts, embeddings=embeddings, ids=ids)

    def query(self, query_embedding: list[float], top_k: int = 3) -> list[str]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results["documents"][0]
