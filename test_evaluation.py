"""
Chapter 2 — Evaluation & Guardrails

Uses RAGAS (Retrieval-Augmented Generation Assessment Suite) to score
the pipeline's answers, following the approach in Krish Naik's RAG
evaluation crash course (github.com/krishnaik06/RAG-Tutorials).

RAGAS defaults to OpenAI for its internal judge LLM and embeddings, which
this project doesn't have quota for. Both are explicitly overridden here:
- judge LLM -> Groq (same provider already used elsewhere in this repo)
- embeddings -> the same local sentence-transformers model already used
  for retrieval, so no new API dependency is introduced

Two metrics that don't require hand-written ground-truth answers:
- faithfulness:      does the answer stay grounded in the retrieved context?
                      (this is a FAITHFULNESS check, not a factuality check —
                      see notes/research_notes.md, FRANQ section, for why
                      that distinction matters)
- answer_relevancy:  does the answer actually address the question asked?

Logs per-question scores to a CSV so failures are traceable after the fact
(Chapter 2 checklist: "can you tell, after the fact, why a bad answer
happened?").
"""

import csv
import os
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.embedders.local_embedder import LocalEmbedder
from src.vectorstores.chroma_vectorstore import ChromaVectorStore
from src.llms.groq_llm import GroqLLM
from src.pipeline import RAGPipeline

load_dotenv()

# One question per source file, matching content confirmed via
# inspect_true_data.py, plus one out-of-scope check.
TEST_QUESTIONS = [
    "What is prohibited under AICertify's AI Ethics and Responsible Conduct Policy regarding open-source models?",  # PDF
    "Should black-box machine learning models interact directly with production systems, according to the compliance framework?",  # TXT
    "What does the AICertify platform protect against?",  # PPTX
    "How does the platform mitigate false positives during automated bias evaluations?",  # HTML
    "What is required for high-availability deployment of the aicertify-agent platform?",  # DOCX
    "What is the company's pet policy?",  # out-of-scope check, expects "I don't know"
]


def run_pipeline_and_collect(pipeline: RAGPipeline, questions: list[str]) -> dict:
    data = {"question": [], "answer": [], "contexts": []}

    for question in questions:
        query_embedding = pipeline.embedder.embed(question)
        retrieved_chunks = pipeline.vectorstore.query(query_embedding, top_k=3)
        answer = pipeline.query(question)

        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(retrieved_chunks)

    return data


def log_results_to_csv(df, path: str = "evaluation_log.csv"):
    df.to_csv(path, index=False)
    print(f"Logged results to {path}")


if __name__ == "__main__":
    pipeline = RAGPipeline(
        embedder=LocalEmbedder(),
        vectorstore=ChromaVectorStore(),
        llm=GroqLLM(),
    )
    count = pipeline.ingest_directory("data/sample_docs/true_data")
    print(f"Ingested {count} chunks.\n")

    raw_data = run_pipeline_and_collect(pipeline, TEST_QUESTIONS)
    eval_dataset = Dataset.from_dict(raw_data)

    # RAGAS's judge LLM and embeddings - overriding the OpenAI defaults
    judge_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.1-8b-instant"))
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # answer_relevancy normally generates several candidate questions per
    # answer in one API call (n > 1) to score similarity. Groq's API
    # rejects n > 1, so strictness is capped at 1 (one candidate question
    # per answer instead of several) to keep this metric working.
    answer_relevancy_metric = AnswerRelevancy(strictness=1)

    print("Running RAGAS evaluation (calls Groq several times per question)...")
    result = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy_metric],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\n--- RAGAS Scores (mean across all questions) ---")
    print(result)

    log_results_to_csv(result.to_pandas())