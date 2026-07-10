from src.embedders.local_embedder import LocalEmbedder
from src.vectorstores.chroma_vectorstore import ChromaVectorStore
from src.llms.groq_llm import GroqLLM
from src.pipeline import RAGPipeline
from src.tools.rag_query_tool import RAGQueryTool
from src.tools.calculator_tool import CalculatorTool
from src.agent import Agent

llm = GroqLLM()
pipeline = RAGPipeline(embedder=LocalEmbedder(), vectorstore=ChromaVectorStore(), llm=llm)
pipeline.ingest_directory("data/sample_docs")

agent = Agent(llm=llm, tools=[RAGQueryTool(pipeline), CalculatorTool()])

# Should trigger the calculator tool, not RAG
print(agent.run("What is 85 times 30?"))
print()
# Should trigger the RAG tool
print(agent.run("What is the daily meal expense limit?"))