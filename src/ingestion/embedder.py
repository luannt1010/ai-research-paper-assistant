from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings


class DocumentEmbedding:
    def __init__(self, model_name: str = "qwen3-embedding:8b", dims: int = 1024, num_gpu: int = 1):
        self.model = OllamaEmbeddings(model=model_name, dimensions=dims,
                                      validate_model_on_init=True,
                                      num_ctx=8192, num_gpu=num_gpu)
        # self.model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_docs(self, chunks: list[Document]) -> List[List[float]]:
        chunks = [chunk.page_content for chunk in chunks]
        return self.model.embed_documents(chunks)

    def embed_query(self, query: str) -> List[float]:
        return self.model.embed_query(query)

# loader = SimpleLoader()
# splitter = RecursiveChunk(chunk_size=1200, chunk_overlap=50)
# all_docs = loader.load_dir(r"D:\private\ai-research-paper-assistant\papers")
# chunks = splitter.split(all_docs)
#
# model = DocumentEmbedding()
# res = model.embed_docs(chunks)
# print(res)
# print(len(res))
