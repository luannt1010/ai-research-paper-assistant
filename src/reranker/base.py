from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

class BaseReRanker(ABC):

    @abstractmethod
    def rerank(self, query: str, documents: List[dict], top_k: int = 5) -> List[dict]:
        pass

