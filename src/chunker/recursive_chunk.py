from .base import BaseChunker
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

class RecursiveChunk(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=SEPARATOR)

    def split(self, documents: list[Document]) -> list[Document]:
        return self._enrich(self.splitter.split_documents(documents))