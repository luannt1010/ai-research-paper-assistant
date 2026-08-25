from typing import List
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class ReRanker:
    def __init__(self):
        self.model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    def _compute_score(self, query: str, documents: List[dict]) -> List[float]:
        if not documents or len(documents) == 0:
            raise ValueError("Documents are empty!")
        text_pairs = [(query, doc["content"]) for doc in documents]
        return self.model.score(text_pairs)

    def rerank(self, query: str, documents: List[dict], top_k: int = 5) -> List[dict]:
        scores = self._compute_score(query, documents)
        for doc, score in zip(documents, scores):
            doc["cross_encoder_score"] = score
        documents.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
        return documents[:top_k]

# documents = [
#     {
#         "id": 1,
#         "document_id": "paper1.pdf",
#         "content": "ArcFace uses additive angular margin for face recognition.",
#         "metadata": {}
#     },
#     {
#         "id": 2,
#         "document_id": "paper1.pdf",
#         "content": "ResNet is a convolutional neural network.",
#         "metadata": {}
#     },
#     {
#         "id": 3,
#         "document_id": "paper2.pdf",
#         "content": "ArcFace improves face recognition performance.",
#         "metadata": {}
#     },
#     {
#         "id": 4,
#         "document_id": "paper3.pdf",
#         "content": "Transformer models use self attention.",
#         "metadata": {}
#     }
# ]
#
# query = "What is arcface?"
#
# r = ReRanker()
#
# scores = r.rerank(query, documents)
# print(scores)