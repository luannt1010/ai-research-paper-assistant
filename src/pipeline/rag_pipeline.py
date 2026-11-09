from src.utils.helpers import format_context
from src.retriever import HybridRetriever
from src.generation import prompt
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


class RAGChain:
    def __init__(self, llm, repository, embedder, reranker, *, rrf_k=10, rerank_k=5):
        if not 1 <= rerank_k <= rrf_k:
            raise ValueError("Expected 1 <= rerank_k <= rrf_k.")
        self.rrf_k = rrf_k
        self.rerank_k = rerank_k
        self.llm = llm
        self.reranker = reranker
        self.prompt = prompt

        self.hybrid_searcher = HybridRetriever(
            embedder=embedder,
            repository=repository,
        )
        self.refresh_index()

        self._chain = self.get_chain()

    def refresh_index(self) -> None:
        self.hybrid_searcher.fit()

    def _retrieve_context(self, inputs: dict) -> dict:
        question = inputs["question"]
        docs = self.hybrid_searcher.retrieve(
            question,
            top_k=inputs["rrf_k"],
        )

        if docs:
            docs = self.reranker.rerank(
                question,
                docs,
                top_k=min(inputs["rerank_k"], len(docs)),
            )

        return {
            "question": question,
            "context": format_context(docs) if docs else "",
        }

    def get_chain(self):
        return (
            RunnableLambda(self._retrieve_context)
            | self.prompt
            | RunnableLambda(self.llm.generate)
            | StrOutputParser()
        )

    def generate(
        self,
        question: str,
        rrf_k: int | None = None,
        rerank_k: int | None = None,
    ) -> str:
        question = question.strip()
        rrf_k = self.rrf_k if rrf_k is None else rrf_k
        rerank_k = self.rerank_k if rerank_k is None else rerank_k

        if not question:
            raise ValueError("question không được để trống.")

        if not 1 <= rerank_k <= rrf_k:
            raise ValueError("Cần đảm bảo 1 <= rerank_k <= rrf_k.")

        return self._chain.invoke({
            "question": question,
            "rrf_k": rrf_k,
            "rerank_k": rerank_k,
        })
