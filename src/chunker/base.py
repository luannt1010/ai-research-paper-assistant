from __future__ import annotations

from abc import ABC, abstractmethod

from typing import List
from langchain_core.documents import Document

class BaseChunker(ABC):

    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]:
        pass
