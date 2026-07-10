from src.embedders.local_embedder import LocalEmbedder
from src.vectorstores.chroma_vectorstore import ChromaVectorStore
from src.llms.groq_llm import GroqLLM
from src.rerankers.flashrank_reranker import FlashRankReranker
from src.pipeline import RAGPipeline

pipeline = RAGPipeline(
    embedder=LocalEmbedder(),
    vectorstore=ChromaVectorStore(),
    llm=GroqLLM(),
)
count = pipeline.ingest_directory("data/sample_docs")
print(f"Ingested {count} chunks.\n")

question = "What is the maximum coverage limit?"

query_embedding = pipeline.embedder.embed(question)

# Retrieve a wider pool, same as the pipeline does internally when a reranker is set
candidates = pipeline.vectorstore.query(query_embedding, top_k=5)

print("--- Plain retrieval order (by embedding similarity) ---")
for i, chunk in enumerate(candidates):
    print(f"[{i}] {chunk[:150]}...\n")

reranker = FlashRankReranker()
reranked = reranker.rerank(question, candidates, top_k=3)

print("--- After reranking (top 3) ---")
for i, chunk in enumerate(reranked):
    print(f"[{i}] {chunk[:150]}...\n")
