import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict

from ragas.llms import llm_factory, LangchainLLMWrapper
from ragas.embeddings.base import embedding_factory, LangchainEmbeddingsWrapper
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
    Faithfulness,
    FactualCorrectness
)

from src.generation import OllamaLLM
from src.embedder import OllamaEmbedder

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

def precision_recall_at_k(docs_true: List[List[str]], queries: List[List[str]], top_k: int = 5) -> Tuple[float, float]:
    """
    P@k = |relevant documents in top k| / k
    R@k = |relevant documents in top k| / |total relevant documents|
    """
    if top_k == 0 or top_k < 0:
        raise ValueError("Top k muse be greater than 0")
    r, p = [], []
    for query, doc_true in zip(queries, docs_true):
        q_top_k = query[:top_k]
        tmp = len([doc for doc in q_top_k if doc in doc_true])
        r.append(tmp / len(doc_true) if doc_true else 0)
        p.append(tmp / top_k)
    return (np.mean(p), np.mean(r))

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

def create_evaluator(llm_model_name: str = "llama3.1", embed_model_name: str = "embeddinggemma:300m"):
    # client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    # llm = llm_factory(model=llm_model_name, client=client, provider="openai", temperature=0, max_tokens=8192)
    # embedder = embedding_factory(provider="openai", model=embed_model_name, client=client)
    # print("Load evaluator OK")
    llm = OllamaLLM(model_name=llm_model_name, temperature=0, num_gpu=1, num_ctx=8192, reasoning=False).generator
    embedder = OllamaEmbedder(model_name=embed_model_name, dimensions=2046, num_ctx=8192, num_gpu=1).embedder
    return llm, embedder

def compute_mean_gen_results(results: pd.DataFrame) -> Dict[str, float]:
    numeric_df = results.select_dtypes(include=["float", "int"])
    results = {}
    for col in numeric_df.columns:
        results[col] = numeric_df[col].mean()
    return results

async def evaluate_context_gen_quality(llm, embedder, eval_data: List[dict]) -> pd.DataFrame:

    context_p = ContextPrecision(llm=llm)
    context_r = ContextRecall(llm=llm)
    faithfulness = Faithfulness(llm=llm)
    ans_relevancy = AnswerRelevancy(llm=llm, embeddings=embedder)
    factual_correctness = FactualCorrectness(llm=llm)

    results = []

    for sample in tqdm(eval_data, desc="Evaluating"):

        start = time.perf_counter()

        user_input = sample["user_input"]
        reference = sample["reference"]
        retrieved_contexts = sample["retrieved_contexts"]
        response = sample["response"]

        p_score = await context_p.ascore(user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference)
        r_score = await context_r.ascore(user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference)
        faithfulness_score = await faithfulness.ascore(user_input=user_input, retrieved_contexts=retrieved_contexts, response=response)
        ans_relevancy_score = await ans_relevancy.ascore(user_input=user_input, response=response)
        factual_correctness_score = await factual_correctness.ascore(reference=reference, response=response)

        end = time.perf_counter()

        results.append({
            "user_input": user_input,
            "reference": reference,
            "response": response,
            "retrieved_contexts": retrieved_contexts,

            "context_precision": float(p_score.value),
            "context_recall": float(r_score.value),

            "faithfulness": float(faithfulness_score.value),
            "answer_relevancy": float(ans_relevancy_score.value),
            "factual_correctness": float(factual_correctness_score.value),

            "Time": (end-start)/60
        })

    return pd.DataFrame(results)




