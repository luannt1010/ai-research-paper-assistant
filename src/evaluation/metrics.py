import numpy as np
from typing import List

def hit_rate_at_k(docs_true: List[List[str]], queries: List[List[str]], top_k: int = 5) -> float:
    """
    Hit rate = Number of Queries with at least one relevant document retrieved / Total number of queries
    """
    if top_k == 0 or top_k < 0:
        raise ValueError("Top k muse be greater than 0")
    hits = 0
    for query, doc_true in zip(queries, docs_true):
        q_top_k = query[:top_k]
        if any(q in doc_true for q in q_top_k):
            hits += 1
    return hits / len(docs_true)

def precision_at_k(docs_true: List[List[str]], queries: List[List[str]], top_k: int = 5) -> float:
    """
    P@k = |relevant documents in top k| / k
    """
    if top_k == 0 or top_k < 0:
        raise ValueError("Top k muse be greater than 0")
    p = []
    for query, doc_true in zip(queries, docs_true):
        q_top_k = query[:top_k]
        set_doc_true = set(doc_true)
        hits = len([doc for doc in q_top_k if doc in set_doc_true])
        p.append(hits / top_k)
    return np.mean(p)

def recall_at_k(docs_true: List[List[str]], queries: List[List[str]], top_k: int = 5) -> float:
    """
        P@k = |relevant documents in top k| / k
    """
    if top_k == 0 or top_k < 0:
        raise ValueError("Top k muse be greater than 0")
    r = []
    for query, doc_true in zip(queries, docs_true):
        q_top_k = query[:top_k]
        set_doc_true = set(doc_true)
        hits = len([doc for doc in q_top_k if doc in set_doc_true])
        r.append(hits / len(doc_true) if doc_true else 0)
    return np.mean(r)

def mean_reciprocal_rank(docs_true: List[List[str]], queries: List[List[str]]) -> float:
    """
    MRR = (1 / N) * Σ (1 / rank_i)
    Where:
        N: total number of queries
        rank_i: rank position of the first relevant document for the i-th query
    """
    reciprocal_ranks = []
    for query, doc_true in zip(queries, docs_true):
        rr = 0
        for rank, doc in enumerate(query, start=1):
            if doc in doc_true:
                rr = 1 / rank
                break
        reciprocal_ranks.append(rr)
    return np.mean(reciprocal_ranks)

def mean_average_precision(docs_true: List[List[str]], queries: List[List[str]]) -> float:
    """
    MAP = (1 / N) * Σ AP_i
    AP_i = (1 / R_i) * Σ (P_i(k) * rel_i(k))
    Where:
        N: total number of queries
        AP_i: average precision for the i-th query
        R_i: number of relevant documents for query i
        P_i(k): precision at cutoff k
        rel_i(k): 1 if the document at rank k is relevant, else 0
    """
    ap = []
    for query, doc_true in zip(queries, docs_true):
        hits = 0
        p = 0
        for rank, doc in enumerate(query, start=1):
            if doc in doc_true:
                hits += 1
                p += hits / rank
        ap.append(p / len(doc_true) if doc_true else 0)
    return np.mean(ap)

def ndcg_at_k(docs_true: List[List[str]], queries: List[List[str]], top_k: int = 5) -> float:
    """
    nDCG_p = DCG_p / IDCG_p
    DCG_p = Σ ((2^rel_i - 1) / log2(i + 1))
    IDCG_p = Σ ((2^rel_ideal_i - 1) / log2(i + 1))
    A retrieved document is considered:
        relevant     -> relevance score = 1
        non-relevant -> relevance score = 0
    Where:
        p: rank position cutoff
        rel_i: relevance score of the document at rank i
        rel_ideal_i: relevance of document at rank i in ideal ordering
    """
    if top_k == 0 or top_k < 0:
        raise ValueError("Top k muse be greater than 0")
    ndcg_scores = []
    for query, doc_true in zip(queries, docs_true):
        q_top_k = query[:top_k]
        dcg = sum([1 / np.log2(idx + 1) if doc in doc_true else 0 for idx, doc in enumerate(q_top_k, start=1)])
        ideal_docs_k = doc_true[:top_k]
        idcg = sum([1 / np.log2(idx + 1) for idx, _ in enumerate(ideal_docs_k, start=1)])
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(ndcg_scores)


