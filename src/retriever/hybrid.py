from typing import List
from src.database import PGVectorStore
from src.database import QdrantVectorStore
from src.embedder import OllamaEmbedder
from .dense import DenseRetriever
from .bm25 import BM25
from .base import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, embedder: OllamaEmbedder, repository: PGVectorStore | QdrantVectorStore):

        self.bm25 = BM25()
        self.repo = repository
        self.dense_retriever = DenseRetriever(embedder, repository)

    def fit(self) -> None:
        all_chunks = self.repo.get_all_chunks()
        self.bm25.fit(all_chunks)

    def _rrf(self, bm25_res: List[dict], dense_res: List[dict],
            k: int = 60, top_k: int = 5) -> List[dict]:
        mapping = {}

        # BM25 ranking
        for rank, doc in enumerate(bm25_res, start=1):
            chunk_id = doc["id"]
            if chunk_id not in mapping:
                mapping[chunk_id] = {"doc": doc, "score": 0.0}
            mapping[chunk_id]["score"] += 1 / (k + rank)

        # Dense ranking
        for rank, doc in enumerate(dense_res, start=1):
            chunk_id = doc["id"]
            if chunk_id not in mapping:
                mapping[chunk_id] = {"doc": doc, "score": 0.0}
            mapping[chunk_id]["score"] += 1 / (k + rank)

        results = []

        for chunk_id, item in mapping.items():
            doc = item["doc"]
            results.append({
                "id": doc["id"],
                "document_id": doc["document_id"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "rrf_score": item["score"]
            })

        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results[:top_k]

    def retrieve(self, query: str, top_k: int=5) -> List[dict]:

        dense_results = self.dense_retriever.retrieve(query, 20)

        spare_results = self.bm25.retrieve(query, 20)

        return self._rrf(spare_results, dense_results, top_k=top_k)

