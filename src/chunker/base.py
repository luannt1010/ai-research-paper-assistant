from __future__ import annotations

from abc import ABC, abstractmethod

from typing import List
from langchain_core.documents import Document

class BaseChunker(ABC):

    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]:
        pass

    @staticmethod
    def _enrich(documents: List[Document]) -> List[Document]:
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "chunk_id": i,
                "length_chunk": len(doc.page_content)
            })
        return documents