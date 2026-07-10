from src.tools.base_tool import BaseTool
from src.pipeline import RAGPipeline


class RAGQueryTool(BaseTool):
    """
    Wraps the existing Chapter 1 RAGPipeline as a callable tool.
    The agent doesn't know or care that this is RAG under the hood —
    it just sees a tool with a name, a description, and a run() method.
    """

    name = "rag_query"
    description = (
        "Use this to answer questions about company policy, product FAQs, "
        "or technical guides using the internal document knowledge base. "
        "Input should be a natural language question."
    )

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def run(self, input_str: str) -> str:
        return self.pipeline.query(input_str)