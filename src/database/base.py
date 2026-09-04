from __future__ import annotations

from abc import ABC, abstractmethod

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

class BaseVectorStore(ABC):

    @abstractmethod
    def _get_connection(self):
        pass

    @abstractmethod
    def insert(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]]=None) -> List[dict]:
        pass

    @abstractmethod
    def delete(self, filters: Optional[Dict[str, Any]]=None) -> None:
        pass

    @abstractmethod
    def count(self, filters: Optional[Dict[str, Any]]=None) -> int:
        pass

    @abstractmethod
    def get_all_chunks(self) -> List[dict]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

