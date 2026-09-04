from typing import List
from src.database import PGVectorStore
from src.database import QdrantVectorStore
from src.embedder import OllamaEmbedder
from .base import BaseRetriever

class DenseRetriever(BaseRetriever):
    def __init__(self, embedder: OllamaEmbedder, repository: PGVectorStore | QdrantVectorStore):
        self.embedder = embedder
        self.repository = repository

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        query_embedding = self.embedder.embed_query(query)

        results = self.repository.search(query_embedding=query_embedding, top_k=top_k)

        return results