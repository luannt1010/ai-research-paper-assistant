from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

SEPARATOR = [
    "\n#{1,6} ",
    "```\n",
    "\n\*\*\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

class RecursiveChunk:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                       add_start_index=True, separators=SEPARATOR)
    def split(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)

class SematicChunk:
    def __init__(self, embedding_model, threshold: float):
        self.splitter = SemanticChunker(embeddings=embedding_model,
                                        breakpoint_threshold_amount=threshold)
    def split(self, documents) -> list[Document]:
        return self.splitter.split_documents(documents)


