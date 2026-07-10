from src.embedders.local_embedder import LocalEmbedder
from src.vectorstores.chroma_vectorstore import ChromaVectorStore
from src.llms.groq_llm import GroqLLM
from src.pipeline import RAGPipeline

pipeline = RAGPipeline(
    embedder=LocalEmbedder(),
    vectorstore=ChromaVectorStore(),
    llm=GroqLLM()
)

count = pipeline.ingest_directory("data/sample_docs/true_data")
print(f"Ingested {count} chunks.\n")

# Check 1: question with no answer in any doc — should say "I don't know", not hallucinate
print("--- Check 1: Out-of-scope question ---")
answer1 = pipeline.query("What is the company's pet policy?")
print("Answer:", answer1, "\n")

# Check 2: question that should pull from product_faq.txt or technical_guide.txt,
# not company_policy.txt — proves retrieval isn't just defaulting to one file
print("--- Check 2: Cross-document retrieval ---")
answer2 = pipeline.query("PUT_A_QUESTION_FROM_YOUR_FAQ_OR_TECH_GUIDE_HERE")
print("Answer:", answer2, "\n")