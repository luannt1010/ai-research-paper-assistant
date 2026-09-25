import time
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.retriever import BaseRetriever
from src.database import BaseVectorStore
from src.reranker import BaseReRanker
from src.embedder import BaseEmbedder
from .metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    mean_average_precision
)

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
    Faithfulness,
    FactualCorrectness
)

class Evaluator:
    def __init__(
            self,
            repository: BaseVectorStore,
            embedder: BaseEmbedder,
            retriever: BaseRetriever,
            reranker: BaseReRanker | None = None,
            eval_llm_model_name="gemini-3.6-flash",
            eval_embedder_model_name="gemini-embedding-001",
            **kwargs):

        self.kwargs = kwargs

        self.repo = repository
        self.embedder = embedder
        self.reranker = reranker
        self.retriever = retriever

        self.eval_llm_model_name = eval_llm_model_name
        self.eval_embedder_model_name = eval_embedder_model_name
        self.eval_llm, self.eval_embedder = self._create_evaluator()
        print("Init Successfully!")

    def _create_evaluator(self):
        client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
        llm = llm_factory(model=self.eval_llm_model_name, client=client, provider="openai", **self.kwargs)
        embedder = embedding_factory(provider="openai", model=self.eval_embedder_model_name, client=client)
        print("Create evaluator successfully!")
        return llm, embedder

    def _make_retrieve_ids(
            self,
            queries: Dict[str, str],
            qrels: Dict[str, List[str]],
            candidate_k: int | None = None,
            top_k: int = 5) -> Tuple[List[List[str]], List[List[str]]]:
        predict_chunks_ids = []
        gt_chunks_ids = []
        for q_id, text in tqdm(queries.items(), desc="PreparingIDS"):
            if q_id in qrels:
                gt_chunks_ids.append(qrels[q_id])
                if self.reranker is not None:
                    if candidate_k is None:
                        raise ValueError("candidate_k is none value.")
                    searched = self.retriever.retrieve(text, candidate_k)
                    searched = self.reranker.rerank(text, searched, top_k)
                else:
                    searched = self.retriever.retrieve(text, top_k)
                ids = []
                for doc in searched:
                    ids.append(doc["document_id"])
                predict_chunks_ids.append(ids)
        return predict_chunks_ids, gt_chunks_ids

    async def _evaluate_context_gen_quality(
            self,
            eval_data: List[dict],
            metrics: List[str]) -> pd.DataFrame:

        if not metrics:
            return pd.DataFrame([{}])
        metrics = [metric.lower() for metric in metrics]
        metric_objects = {}
        if "context_precision" in metrics:
            metric_objects["context_precision"] = ContextPrecision(llm=self.eval_llm)
        if "context_recall" in metrics:
            metric_objects["context_recall"] = ContextRecall(llm=self.eval_llm)
        if "faithfulness" in metrics:
            metric_objects["faithfulness"] = Faithfulness(llm=self.eval_llm)
        if "answer_relevancy" in metrics:
            metric_objects["answer_relevancy"] = AnswerRelevancy(llm=self.eval_llm, embeddings=self.eval_embedder)
        if "factual_correctness" in metrics:
            metric_objects["factual_correctness"] = FactualCorrectness(llm=self.eval_llm)

        results = []

        for sample in tqdm(eval_data, desc="Evaluating"):

            start = time.perf_counter()
            result = {}

            user_input = sample["user_input"]
            reference = sample["reference"]
            retrieved_contexts = sample["retrieved_contexts"]
            response = sample["response"]

            for name, metric in metric_objects.items():
                if name == "context_precision" or name == "context_recall":
                    score = await metric.ascore(user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference)
                elif name == "faithfulness":
                    score = await metric.ascore(user_input=user_input, retrieved_contexts=retrieved_contexts, response=response)
                elif name == "answer_relevancy":
                    score = await metric.ascore(user_input=user_input, response=response)
                elif name == "factual_correctness":
                    score = await metric.ascore(reference=reference, response=response)
                result[name] = float(score.value)

            end = time.perf_counter()

            result["Time"] = (end - start) / 60
            result["user_input"] = user_input
            result["reference"] = reference
            result["response"] = response
            result["retrieved_contexts"] = retrieved_contexts

            results.append(result)

        return pd.DataFrame(results)

    def _evaluate_retrieve_quality(self,
            queries: Dict[str, str],
            qrels: Dict[str, List[str]],
            retrieve_metrics: List[str],
            candidate_k: int | None = None,
            top_k: int = 5) -> pd.DataFrame:

        if not retrieve_metrics:
            return pd.DataFrame([{}])
        retrieve_metrics = [metric.lower() for metric in retrieve_metrics]
        predict_chunks_ids, gt_chunks_ids = self._make_retrieve_ids(queries, qrels, candidate_k, top_k)

        retrieve_results = {}
        if "precision" in retrieve_metrics:
            precision = precision_at_k(gt_chunks_ids, predict_chunks_ids, top_k)
            retrieve_results[f"precision@{top_k}"] = precision
        if "recall" in retrieve_metrics:
            recall = recall_at_k(gt_chunks_ids, predict_chunks_ids, top_k)
            retrieve_results[f"recall@{top_k}"] = recall
        if "hit" in retrieve_metrics:
            hit_score = hit_rate_at_k(gt_chunks_ids, predict_chunks_ids, top_k)
            retrieve_results[f"hit@{top_k}"] = hit_score
        if "mrr" in retrieve_metrics or "mean_reciprocal_rank" in retrieve_metrics:
            mrr_score = mean_reciprocal_rank(gt_chunks_ids, predict_chunks_ids)
            retrieve_results[f"mrr@{top_k}"] = mrr_score
        if "map" in retrieve_metrics or "mean_average_precision" in retrieve_metrics:
            map_score = mean_average_precision(gt_chunks_ids, predict_chunks_ids)
            retrieve_results[f"map@{top_k}"] = map_score
        if "ndcg" in retrieve_metrics:
            ndcg_score = ndcg_at_k(gt_chunks_ids, predict_chunks_ids, top_k)
            retrieve_results[f"ndcg@{top_k}"] = ndcg_score

        return pd.DataFrame([retrieve_results])

    async def evaluate(
            self,
            queries: Dict[str, str],
            qrels: Dict[str, List[str]],
            eval_context_gen_data: List[dict],
            context_gen_metrics: List[str],
            retrieve_metrics: List[str],
            candidate_k: int | None = None,
            top_k: int = 5) -> Dict[str, pd.DataFrame]:

        retrieve_results = self._evaluate_retrieve_quality(queries, qrels, retrieve_metrics, candidate_k, top_k)
        context_gen_results = await self._evaluate_context_gen_quality(eval_context_gen_data, context_gen_metrics)

        print("Evaluating progress finished successfully!")

        return {
            "retrieve_quality": retrieve_results,
            "context_gen_quality": context_gen_results
        }





        
        
