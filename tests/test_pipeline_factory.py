from copy import deepcopy
from unittest.mock import Mock

import pytest
import yaml
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.pipeline import PipelineFactory
from src.pipeline.factory import DEFAULT_CONFIG_PATH


@pytest.fixture
def config():
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def make_factory(tmp_path, config, repository=None):
    path = tmp_path / "app.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return PipelineFactory(path, repository=repository)


@pytest.mark.parametrize("section,key,value", [
    ("embedding", "provider", "unknown"),
    ("llm", "model_name", ""),
    ("chunking", "length_unit", "token"),
    ("chunking", "chunk_overlap", 1200),
    ("chunking", "strategy", "markdown"),
    ("retrieval", "rrf_k", True),
    ("retrieval", "rerank_k", 11),
    ("reranker", "batch_size", 0),
    ("llm", "options", {"temperature": 1}),
    ("chunking", "typo", 1),
])
def test_invalid_config_fails_before_building_components(tmp_path, config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        make_factory(tmp_path, config)


def test_config_loading_is_lazy_and_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = PipelineFactory()
    assert not {"repository", "embedder", "llm", "reranker", "chunker"} & factory.__dict__.keys()
    factory.close()  # Must not open a database just to close it.


def test_ingestion_and_rag_share_embedding_and_repository(tmp_path, config):
    repository = Mock()
    repository.get_all_chunks.return_value = []
    factory = make_factory(tmp_path, config, repository)
    ingestion = factory.get_ingestion_pipeline()
    rag = factory.get_rag_pipeline()
    assert ingestion.repo is repository
    assert rag.hybrid_searcher.repo is repository
    assert ingestion.embedder is factory.embedder
    assert factory.embedder is factory.embedder
    assert factory.llm.model_name == config["llm"]["model_name"]
    assert factory.embedder.kwargs == config["embedding"]["options"]
    assert factory.reranker.model is None
    assert factory.llm._model is None
    factory.close()
    repository.close.assert_not_called()


def test_rag_uses_yaml_defaults_and_per_call_overrides(tmp_path, config):
    config["retrieval"] = {"rrf_k": 8, "rerank_k": 2}
    repository = Mock()
    repository.get_all_chunks.return_value = []
    factory = make_factory(tmp_path, config, repository)
    factory.llm = Mock()
    factory.llm.generate.return_value = AIMessage(content="Answer")
    factory.reranker = Mock()
    docs = [{"content": "Evidence", "metadata": {"source": "paper.pdf", "page": 1}}] * 3
    factory.reranker.rerank.return_value = docs[:2]
    rag = factory.get_rag_pipeline()
    rag.hybrid_searcher.retrieve = Mock(return_value=deepcopy(docs))
    assert rag.generate(" Question ") == "Answer"
    rag.hybrid_searcher.retrieve.assert_called_with("Question", top_k=8)
    assert factory.reranker.rerank.call_args.kwargs["top_k"] == 2
    rag.generate("Question", rrf_k=6, rerank_k=1)
    rag.hybrid_searcher.retrieve.assert_called_with("Question", top_k=6)
    assert factory.reranker.rerank.call_args.kwargs["top_k"] == 1


def test_new_rag_pipeline_refits_after_ingestion(tmp_path, config):
    repository = Mock()
    repository.get_all_chunks.return_value = []
    factory = make_factory(tmp_path, config, repository)
    first = factory.get_rag_pipeline()
    repository.get_all_chunks.return_value = [{"id": 1, "content": "new paper"}]
    second = factory.get_rag_pipeline()
    assert first is not second
    assert second.hybrid_searcher.bm25.num_docs == 1
    assert repository.get_all_chunks.call_count == 2


def test_recursive_chunker_respects_config(tmp_path, config):
    config["chunking"].update(chunk_size=20, chunk_overlap=0)
    factory = make_factory(tmp_path, config)
    chunks = factory.chunker.split([Document(page_content="word " * 30)])
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 20 for chunk in chunks)


def test_owned_repository_closed_once(config):
    factory = PipelineFactory()
    repository = Mock()
    factory.repository = repository
    factory.close()
    factory.close()
    repository.close.assert_called_once()


def test_upload_metadata_reaches_stored_chunks():
    from src.pipeline import IngestionPipeline
    loader, chunker, embedder, repository = Mock(), Mock(), Mock(), Mock()
    loader.load_pdf.return_value = [Document(
        page_content="Evidence", metadata={"page": 3, "source": "temp.pdf"},
    )]
    chunker.split.side_effect = lambda docs: docs
    embedder.embed_documents.return_value = [[0.1, 0.2]]
    pipeline = IngestionPipeline(loader, chunker, embedder, repository)
    metadata = {"document_id": "unique-id", "filename": "paper.pdf", "source": "folder/paper.pdf"}
    assert pipeline.ingest_pdf("temp.pdf", metadata=metadata) == 1
    stored = repository.insert.call_args.args[0][0]
    assert stored.metadata == {"page": 3, **metadata}


def test_qdrant_keeps_chunks_from_consecutive_ingestions():
    from qdrant_client import QdrantClient, models
    from src.database import QdrantVectorStore
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.collection_name = "upload_test"
    store.client = QdrantClient(":memory:")
    try:
        store.client.create_collection(
            store.collection_name,
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )
        for document_id in ("first", "second"):
            store.insert([Document(page_content=document_id, metadata={"document_id": document_id})], [[1.0, 0.0]])
        assert store.count() == 2
        assert {doc["document_id"] for doc in store.get_all_chunks()} == {"first", "second"}
    finally:
        store.close()
