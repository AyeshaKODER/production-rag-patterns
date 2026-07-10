from src.embedders.local_embedder import LocalEmbedder
from src.vectorstores.chroma_vectorstore import ChromaVectorStore
from src.llms.groq_llm import GroqLLM
from src.rerankers.flashrank_reranker import FlashRankReranker
from src.pipeline import RAGPipeline

# Without reranker (old behavior)
pipeline_plain = RAGPipeline(
    embedder=LocalEmbedder(),
    vectorstore=ChromaVectorStore(),
    llm=GroqLLM(),
)
pipeline_plain.ingest_directory("data/sample_docs")

# With reranker
pipeline_reranked = RAGPipeline(
    embedder=LocalEmbedder(),
    vectorstore=ChromaVectorStore(),
    llm=GroqLLM(),
    reranker=FlashRankReranker(),
)
pipeline_reranked.ingest_directory("data/sample_docs")

question = "What is the daily meal expense limit?"

print("--- Without reranker ---")
print(pipeline_plain.query(question))
print()
print("--- With reranker ---")
print(pipeline_reranked.query(question))
