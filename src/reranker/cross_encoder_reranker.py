from typing import List
from .base import BaseReRanker
from sentence_transformers import CrossEncoder

class CrossEncoderReRanker(BaseReRanker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512, batch_size = 32, device: str = "cpu"):
        self.max_length = max_length
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None

    def _load(self):
        if self.model is None:
            self.model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
        return self.model

    def _compute_score(self, query: str, documents: List[dict]) -> List[float]:
        if not documents or len(documents) == 0:
            raise ValueError("Documents are empty!")
        text_pairs = [(query, doc["content"]) for doc in documents]
        return self.model.predict(text_pairs, batch_size=self.batch_size)

    def rerank(self, query: str, documents: List[dict], top_k: int = 5) -> List[dict]:
        self._load()
        scores = self._compute_score(query, documents)
        for doc, score in zip(documents, scores):
            doc["cross_encoder_score"] = score
        documents.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
        return documents[:top_k]