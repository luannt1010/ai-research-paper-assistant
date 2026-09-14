from .base import BaseChunker
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

class SemanticChunk(BaseChunker):
    def __init__(self, embedding_model, threshold: float):
        self.splitter = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_amount=threshold)

    def split(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)