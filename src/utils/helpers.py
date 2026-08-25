from typing import List
from pathlib import Path

def format_context(docs):
    return "\n\n".join(
        f"[Source: {Path(doc["metadata"].get('source')).name}, "
        f"Page: {doc["metadata"].get('page')}]\n"
        f"{doc["content"]}"
        for doc in docs
    )

def rrf(bm25_res: List[dict], dense_res: List[dict],
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

def extract_id_documents(documents: List[dict]) -> List[int]:
    return [doc["id"] for doc in documents]