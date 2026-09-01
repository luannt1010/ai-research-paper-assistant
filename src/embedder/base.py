from __future__ import annotations

from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_ollama.embeddings import Embeddings


class BaseEmbedder(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self._embedder: Embeddings | None = None

    @abstractmethod
    def _build(self) -> Embeddings:
        """Create and return an Embeddings object for the corresponding provider."""

    @property
    def embedder(self) -> Embeddings:
        if self._embedder is None:
            self._embedder = self._build()
        return self._embedder

    def embed_documents(self, chunks: list[Document]) -> List[List[float]]:
        chunks = [chunk.page_content for chunk in chunks]
        return self.embedder.embed_documents(chunks)

    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed_query(query)




