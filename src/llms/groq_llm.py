import os
from groq import Groq
from .base_llm import BaseLLM
from dotenv import load_dotenv
load_dotenv()

class GroqLLM(BaseLLM):
    """
    Groq-backed LLM. Chosen over OpenAI for this prototype because
    Groq's free tier removes API cost as a blocker during development.
    Swapping to OpenAI/Anthropic later means writing one new class
    that implements BaseLLM — pipeline.py does not change.
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: str = None):
        # Falls back to GROQ_API_KEY env var if no key passed explicitly
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except Exception as e:
            # Chapter 1 checklist asks: "what happens when the API call fails?"
            # Answer: it doesn't crash the pipeline silently or return garbage —
            # it fails loudly with context, so the caller can log/retry/handle it.
            raise RuntimeError(f"GroqLLM generation failed: {e}") from e