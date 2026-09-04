from typing import List, Dict
from langchain_core.documents import Document
from pathlib import Path

from src.database.pgvector_storage import PGVectorStore
from src.database.qdrant_storage import QdrantVectorStore
from src.ingestion.parser import SimpleLoader
from src.ingestion.chunker import RecursiveChunk
from src.embedder.ollama_embedder import OllamaEmbedder

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


if __name__ == "__main__":
    # embedder = OllamaEmbedder(dimensions=2046)
    # loader = SimpleLoader()
    # chunker = RecursiveChunk(1200, 100)
    # vector = PGVectorStore()
    # print(vector.count())
    # ingestor = IngestionPipeline(loader, chunker, embedder, vector)
    # ingestor.ingest_pdf(r"D:\private\ai-research-paper-assistant\papers\2401.02385v2.pdf")
    # print(vector.count())
    # vector.close()

    v = PGVectorStore()
    print(v.count())
    v.delete({"document_id": "2401.02385v2"})
    print(v.count())


