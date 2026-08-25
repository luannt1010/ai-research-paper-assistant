from .bm25 import BM25
from typing import List
from src.database.repository import Repository
from src.ingestion.embedder import DocumentEmbedding
from src.utils.helpers import rrf

class HybridSearch:
    def __init__(self, embedder: DocumentEmbedding, repository: Repository, bm25: BM25):
        self.repo = repository
        self.bm25 = bm25
        self.embedder = embedder

    def search(self, query: str, candidate_k: int, top_k: int) -> List[dict]:
        query_embed = self.embedder.embed_query(query)

        dense_results = self.repo.similarity_search(query_embed, candidate_k)

        spare_results = self.bm25.search(query, candidate_k)

        return rrf(spare_results, dense_results, top_k=top_k)

