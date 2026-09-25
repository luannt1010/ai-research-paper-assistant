from typing import List

from src.chunker.base import BaseChunker
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.parser import SimpleLoader

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

class HierarchicalChunker(BaseChunker):
    def __init__(self, parent_chunk_size: int, parent_overlap: int, child_chunk_size: int, child_overlap: int):
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap,
            separators=SEPARATOR,
            add_start_index=True,   # offset trong document gốc, tiện khi debug
            strip_whitespace=True,
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_overlap,
            separators=SEPARATOR,
            # Không bật add_start_index ở mức con: offset sẽ tính từ text của
            # chunk cha chứ không phải document gốc, dễ gây hiểu nhầm. Quan hệ
            # con → cha đã ghi sẵn qua parent_id.
            strip_whitespace=True,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        all_docs = []
        p_id = 0
        for doc in documents:
            parents = self.parent_splitter.split_documents([doc])
            for parent in parents:
                children = self.child_splitter.split_documents([parent])
                for child in children:
                    child.metadata.update({
                        "chunk_level": "child",
                        "p_id": p_id
                    })
                parent.metadata.update({
                    "chunk_level": "parent",
                    "p_id": p_id
                })
                p_id += 1
                all_docs.extend(children)
            all_docs.extend(parents)
        return self._enrich(all_docs)


if __name__ == "__main__":
    chunker = HierarchicalChunker(1200, 50, 300, 0)
    loader = SimpleLoader()
    pdf_file = r"D:\private\ai-research-paper-assistant\papers\1706.03762v7.pdf"

    all_docs = chunker.split(loader.load_pdf(pdf_file))

    print(all_docs)