import pandas as pd
from typing import List, Dict

from src.retriever import HybridRetriever
from src.retriever import DenseRetriever
from .metrics import (
    hit_rate_at_k,
    precision_recall_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    evaluate_context_gen_quality,
    create_evaluator)
from src.utils.helpers import make_retrieve_ids

class Evaluator:
    def __init__( self, repository, embedder, reranker=None, eval_llm_model_name="llama3.1", eval_embedder_model_name="embeddinggemma:300m", search_mode="hybrid"):

        self.search_mode = search_mode
        self.repo = repository
        self.embedder = embedder
        self.reranker = reranker

        self.eval_llm, self.eval_embedder = create_evaluator(
            llm_model_name=eval_llm_model_name,
            embed_model_name=eval_embedder_model_name)
        self.retriever = self._load_retriever()
        print("Init Successfully!")
        
    def _load_retriever(self):
        if self.search_mode.lower() == "dense":
            return DenseRetriever(embedder=self.embedder, repository=self.repo)
        return HybridRetriever(embedder=self.embedder, repository=self.repo)


    async def evaluate(self, queries: Dict[str, str], qrels: Dict[str, List[str]], eval_context_gen_data: List[dict], top_k: int = 5) -> Dict[str, pd.DataFrame]:

        predict_chunks_ids, gt_chunks_ids = make_retrieve_ids(queries, qrels, self.retriever, self.reranker, top_k)
        print("Ok")
        context_gen_results = await evaluate_context_gen_quality(self.eval_llm, self.eval_embedder, eval_context_gen_data)

        
        precision, recall = precision_recall_at_k(gt_chunks_ids, predict_chunks_ids, top_k)
        hit_score = hit_rate_at_k(gt_chunks_ids, predict_chunks_ids, top_k)
        mrr_score = mean_reciprocal_rank(gt_chunks_ids, predict_chunks_ids)
        map_score = mean_average_precision(gt_chunks_ids, predict_chunks_ids)
        ndcg_score = ndcg_at_k(gt_chunks_ids, predict_chunks_ids, top_k)

        retrieve_results = {
            f"precision@{top_k}": precision,
            f"recall@{top_k}": recall,
            f"hit_rate@{top_k}": hit_score,
            f"mrr@{top_k}": mrr_score,
            f"map@{top_k}": map_score,
            f"ndcg@{top_k}": ndcg_score,
        }

        return {
            "retrieve_quality": pd.DataFrame([retrieve_results]),
            "context_gen_quality": context_gen_results
        }





        
        
