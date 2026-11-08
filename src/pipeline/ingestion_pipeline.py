from pathlib import Path
from typing import List
from langchain_core.documents import Document
from src.embedder import BaseEmbedder
from src.database import BaseVectorStore
from src.chunker import BaseChunker

class IngestionPipeline:
    def __init__(self, loader, chunker: BaseChunker, embedder: BaseEmbedder, repository: BaseVectorStore):
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

    def ingest_pdf(self, pdf_path: str, *, metadata: dict | None = None) -> int:
        documents = self.loader.load_pdf(pdf_path)
        if metadata:
            for document in documents:
                document.metadata.update(metadata)
        inserted = self._process(documents)
        print(f"Inserted {inserted} chunks from {Path(pdf_path).name}.")
        return inserted

    def ingest_dir(self, dir_path: str) -> int:
        documents = self.loader.load_dir(dir_path)
        inserted = self._process(documents)
        print(f"Inserted {inserted} chunks from {Path(dir_path).name} folder.")
        return inserted
