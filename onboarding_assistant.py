"""
Chapter 4 — Capstone: New Employee Onboarding Assistant

Business workflow: a new AICertify employee has questions during onboarding
that would normally require pinging HR or a compliance officer. This
assistant answers them automatically — including questions that need both
a policy lookup AND a calculation, which is exactly the case where a plain
RAG pipeline isn't enough and the agent's tool-routing (Chapter 3) earns
its place over Chapter 1 alone.

How Chapters 0-3 combine here:
- Chapter 0 (OOP foundations): swappable embedder/LLM/vectorstore/reranker
  interfaces mean this script doesn't care which concrete providers are
  plugged in underneath RAGPipeline or Agent.
- Chapter 1 (Production RAG): multi-format ingestion (PDF/DOCX/PPTX/HTML/TXT)
  + section-aware chunking + reranking, over the real onboarding document set.
- Chapter 2 (Evaluation): RAGAS faithfulness/relevancy checks (see
  test_evaluation.py) are what would gate these answers before they're
  trusted enough to hand a new employee unsupervised — not run inline here,
  but this is the pipeline that gets evaluated there.
- Chapter 3 (Agent): the ReAct-style bounded agent decides, per question,
  whether it needs the RAG tool, the calculator tool, or both in sequence —
  the onboarding questions below are deliberately mixed to require this.
"""

from src.embedders.local_embedder import LocalEmbedder
from src.vectorstores.chroma_vectorstore import ChromaVectorStore
from src.llms.groq_llm import GroqLLM
from src.rerankers.flashrank_reranker import FlashRankReranker
from src.pipeline import RAGPipeline
from src.tools.rag_query_tool import RAGQueryTool
from src.tools.calculator_tool import CalculatorTool
from src.agent import Agent

llm = GroqLLM()
pipeline = RAGPipeline(
    embedder=LocalEmbedder(),
    vectorstore=ChromaVectorStore(),
    llm=llm,
    reranker=FlashRankReranker(),
)
chunk_count = pipeline.ingest_directory("data/sample_docs/true_data")
print(f"Onboarding knowledge base loaded: {chunk_count} chunks.\n")

agent = Agent(llm=llm, tools=[RAGQueryTool(pipeline), CalculatorTool()])

# Deliberately mixed: pure policy lookups, and one that needs RAG + math
# chained together (agent must look up the reimbursement cap, then
# subtract, in two separate steps of its own reasoning loop).
onboarding_questions = [
    "What is prohibited under AICertify's AI Ethics and Responsible Conduct Policy regarding open-source models?",
    "What is the nightly lodging expense cap for client site travel?",
    "The annual professional development reimbursement limit is a fixed amount, and I've already claimed 45000 of it. Look up the limit and tell me how much I have left.",
]

for question in onboarding_questions:
    print("Q:", question)
    print("A:", agent.run(question))
    print()
