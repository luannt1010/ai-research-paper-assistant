from pathlib import Path
from src.generation import prompt
from src.retriever import HybridRetriever
from langchain_core.documents import Document
from datasets import load_dataset
from collections import defaultdict
from typing import Dict, Tuple, List
from tqdm import tqdm



def format_context(docs):
    return "\n\n".join(
        f"[Source: {Path(doc["metadata"].get('source', '')).name}, "
        f"Page: {doc["metadata"].get('page')}]\n"
        f"{doc["content"]}"
        for doc in docs
    )


def load_scifact():
    """
    Load:
        corpus  : 5183 scientific abstracts
        queries : 1109 queries
        qrels   : relevance judgments

    We use TEST qrels for evaluation.
    """

    corpus_ds = load_dataset(
        "BeIR/scifact",
        "corpus",
        split="corpus"
    )

    queries_ds = load_dataset(
        "BeIR/scifact",
        "queries",
        split="queries"
    )

    qrels_ds = load_dataset(
        "BeIR/scifact-qrels",
        split="test"
    )

    return corpus_ds, queries_ds, qrels_ds

def extract_document_ids(documents: List[dict]) -> List[str]:
    return [str(doc["document_id"]) for doc in documents]

class IngestionPipeline:
    def __init__(self, loader, chunker, embedder, repository):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.repo = repository

    def _process(self, documents: List[Document]) -> int:
        if not documents:
            return 0
        chunks = self.chunker.split(documents)
        if not chunks:
            return 0
        embeddings = self.embedder.embed_documents(chunks)
        if len(chunks) != len(embeddings):
            raise RuntimeError("Chunks and embeddings length mismatch.")

        self.repo.insert(chunks, embeddings)
        return len(chunks)

    def ingest_pdf(self, pdf_path: str) -> int:
        documents = self.loader.load_pdf(pdf_path)
        inserted = self._process(documents)
        print(f"Inserted {inserted} chunks from {Path(pdf_path).name}.")
        return inserted

    def ingest_dir(self, dir_path: str) -> int:
        documents = self.loader.load_dir(dir_path)
        inserted = self._process(documents)
        print(f"Inserted {inserted} chunks from {Path(dir_path).name} folder.")
        return inserted

class RAGChain:
    def __init__(self, llm, repository, embedder, reranker):

        self.repo = repository
        self.embedder = embedder
        self.reranker = reranker
        self.llm = llm

        self.hybrid_searcher = HybridRetriever(embedder=self.embedder, repository=self.repo)
        self.hybrid_searcher.fit()
        self.prompt = prompt


    def chain(self, question: str, rrf_k: int = 10, rerank_k: int = 5) -> str:
        context_rrf = self.hybrid_searcher.retrieve(question, top_k=rrf_k)
        context_reranked = self.reranker.rerank(question, context_rrf, top_k=rerank_k)
        context = format_context(context_reranked)

        message = self.prompt.invoke({"context": context, "question": question})
        response = self.llm.generate(message)

        return response


def build_scifact_data() -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, List[str]]]:
    corpus_ds, queries_ds, qrels_ds = load_scifact()
    corpus = {}
    for row in corpus_ds:
        corpus_id = str(row["_id"])
        corpus[corpus_id] = {
            "title": row.get("title", ""),
            "content": row["text"]
            }

    queries = {}
    for row in queries_ds:
        query_id = str(row["_id"])
        queries[query_id] = row["text"]

    qrels = defaultdict(list)
    for row in qrels_ds:
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        relevance = int(row["score"])
        if relevance > 0:
            qrels[query_id].append(corpus_id)

        # Chỉ evaluate query có GT trong test qrels
    test_queries = {
        query_id: queries[query_id]
        for query_id in qrels
        if query_id in queries
        }

    return corpus, test_queries, dict(qrels)

def make_retrieve_ids(queries: Dict[str, str], qrels: Dict[str, List[str]], retriever, reranker=None, top_k: int = 5) -> Tuple[List[List[str]], List[List[str]]]:
    predict_chunks_ids = []
    gt_chunks_ids = []
    for q_id, text in tqdm(queries.items(), desc="PreparingIDS"):
        if q_id in qrels:
            gt_chunks_ids.append(qrels[q_id])
            searched = retriever.retrieve(text, top_k)
            if reranker is not None:
                searched = reranker.rerank(text, searched, top_k)
            ids = []
            for doc in searched:
                ids.append(doc["document_id"])
            predict_chunks_ids.append(ids)
    return predict_chunks_ids, gt_chunks_ids


