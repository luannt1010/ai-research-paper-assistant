from .base import BaseLLM
from langchain_ollama import ChatOllama

class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str = "qwen3.8:9b", temperature: float = 0.0, **kwargs):
        super().__init__(model_name, temperature, **kwargs)

    def _build(self):
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            **self.kwargs
        )