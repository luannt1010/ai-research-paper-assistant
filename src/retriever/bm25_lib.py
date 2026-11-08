from typing import List
from rank_bm25 import BM25Okapi
import re
from .base import BaseRetriever

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower())
    return tokens

class BM25Lib(BaseRetriever):
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.documents = []
        self.tokenized_documents = []
        self.bm25 = None

    def fit(self, documents: List[dict]) -> None:
        self.documents = documents
        # Tokenize documents
        self.tokenized_documents = [
            tokenize(doc["content"])
            for doc in documents
        ]

        # Initialize Okapi BM25
        self.bm25 = BM25Okapi(
            self.tokenized_documents,
            k1=self.k1,
            b=self.b
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        if not self.documents:
            raise RuntimeError(
                "BM25 has not been fitted. Call fit() before retrieve()."
            )
        # Tokenize query
        query_tokens = query.lower().split()
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        # Sort document indices by score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        # Build results
        results = []
        for idx in ranked_indices[:top_k]:
            doc = self.documents[idx]
            results.append({
                "content": doc["content"],
                "document_id": doc["document_id"],
                "id": doc["id"],
                "metadata": doc["metadata"],
                "bm25_score": float(scores[idx])
            })

        return results