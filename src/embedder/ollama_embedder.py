from src.embedder.base import BaseEmbedder
from langchain_ollama.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "qwen3-embedding:8b", **kwargs):
        super().__init__(model_name, **kwargs)

    def _build(self) -> Embeddings:
        return OllamaEmbeddings(
            model=self.model_name,
            **self.kwargs
        )
