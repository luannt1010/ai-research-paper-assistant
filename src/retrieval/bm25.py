import re
import math
from typing import List
from collections import Counter, defaultdict

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower())
    return tokens

class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.documents = []
        self.doc_length = []
        self.doc_freq = defaultdict(int)
        self.num_docs = 0.0
        self.avg_length = 0.0

    def fit(self, documents: List[dict]) -> None:

        self.documents = documents
        self.num_docs = len(documents)

        self.doc_freq.clear()
        self.doc_length = []

        for doc in self.documents:

            tokens = tokenize(doc["content"])
            doc["tf"] = Counter(tokens)
            self.doc_length.append(len(tokens))

            for token in set(tokens):
                self.doc_freq[token] += 1

        self.avg_length = sum(self.doc_length) / (self.num_docs + 1e-8)

    def _calculate_idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))

    def _score_docs(self, query_tokens: List[str], doc: dict, doc_length: int) -> float:

        tf = doc["tf"]
        score = 0
        for term in query_tokens:
            tf_t = tf.get(term, 0)
            if tf_t == 0:
                continue
            idf = self._calculate_idf(term)
            ratio_doc_len = abs(doc_length) / self.avg_length
            score += (idf * (tf_t * (self.k1 + 1))) / (tf_t + self.k1 * (1 - self.b + self.b * ratio_doc_len))
        return score

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        query_tokens = tokenize(query)
        results = []
        for idx, doc in enumerate(self.documents):
            score = self._score_docs(query_tokens, doc, self.doc_length[idx])
            results.append({"content": doc["content"],
                            "document_id": doc["document_id"],
                            "id": doc["id"],
                            "metadata": doc["metadata"],
                            "bm25_score": score})
        results = sorted(results, key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]

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
# bm25 = BM25()
# bm25.fit(documents)
# results = bm25.search(
#     "ArcFace angular margin",
#     top_k=3
# )
#
# for result in results:
#     print(
#         result["id"],
#         result["bm25_score"],
#         result["content"]
#     )
#
#
# print("arcface:", bm25._calculate_idf("arcface"))
# print("the:", bm25._calculate_idf("the"))
# print("transformer:", bm25._calculate_idf("transformer"))





