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
        pass

    @property
    def embedder(self) -> Embeddings:
        if self._embedder is None:
            self._embedder = self._build()
        return self._embedder

    def embed_documents(self, documents: list[Document], batch_size: int = 32) -> List[List[float]]:
        embeddings = []
        for start in range(0, len(documents), batch_size):
            chunks = [doc.page_content for doc in documents[start:start+batch_size]]
            embedding = self.embedder.embed_documents(chunks)
            embeddings.extend(embedding)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed_query(query)




