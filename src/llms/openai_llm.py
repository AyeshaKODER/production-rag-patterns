from openai import OpenAI
from src.llms.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    """Concrete LLM using OpenAI's chat completion API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
