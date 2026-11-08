from unittest.mock import Mock
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf.errors import PdfReadError

from src.api.main import create_app


@pytest.fixture
def factory():
    factory = Mock()
    factory.config = {"retrieval": {"rrf_k": 12, "rerank_k": 3}}
    factory.repository.count.return_value = 1
    factory.get_ingestion_pipeline.return_value.ingest_pdf.return_value = 7
    factory.get_rag_pipeline.return_value.generate.return_value = "An answer"
    return factory


@pytest.fixture
def client(factory):
    with TestClient(create_app(factory_builder=lambda: factory),
                    raise_server_exceptions=False) as client:
        yield client


def test_lifecycle_and_health_do_not_build_pipelines(factory):
    builder = Mock(return_value=factory)
    app = create_app(factory_builder=builder)
    builder.assert_not_called()
    with TestClient(app) as client:
        builder.assert_called_once()
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
        assert {"/ingestion/ingest", "/chain/generate"} <= schema["paths"].keys()
        factory.get_ingestion_pipeline.assert_not_called()
        factory.get_rag_pipeline.assert_not_called()
        factory.close.assert_not_called()
    factory.close.assert_called_once()


def upload_pdfs(client, *names):
    return client.post("/ingestion/ingest", files=[
        ("files", (name, b"%PDF-1.4\nmock PDF", "application/pdf")) for name in names
    ])


def test_ingest_pdf(client, factory):
    def ingest_file(path, *, metadata):
        assert Path(path).read_bytes().startswith(b"%PDF-")
        assert metadata["filename"] == "paper.pdf"
        return 7
    factory.get_ingestion_pipeline.return_value.ingest_pdf.side_effect = ingest_file
    response = upload_pdfs(client, "paper.pdf")
    assert response.status_code == 200
    body = response.json()
    assert body["successful_files"] == 1
    assert body["inserted_chunks"] == 7
    assert body["files"][0]["filename"] == "paper.pdf"
    saved_path = factory.get_ingestion_pipeline.return_value.ingest_pdf.call_args.args[0]
    assert not Path(saved_path).exists()
    assert not Path(saved_path).parent.exists()


def test_missing_upload(client, factory):
    assert client.post("/ingestion/ingest", json={"path": "paper.pdf"}).status_code == 422
    factory.get_ingestion_pipeline.assert_not_called()


@pytest.mark.parametrize("name,content", [("a.txt", b"%PDF-1.4"), ("a.pdf", b""), ("a.pdf", b"not a pdf")])
def test_invalid_upload(client, factory, name, content):
    response = client.post("/ingestion/ingest", files={"files": (name, content, "application/pdf")})
    assert response.status_code == 200
    assert response.json()["failed_files"] == 1
    factory.get_ingestion_pipeline.assert_not_called()


@pytest.mark.parametrize("error", [
    PdfReadError("bad PDF"), PermissionError("private path"),
    RuntimeError("secret connection string"),
])
def test_ingest_errors_do_not_stop_batch(client, factory, error):
    factory.get_ingestion_pipeline.return_value.ingest_pdf.side_effect = [error, 7]
    response = upload_pdfs(client, "bad.pdf", "good.pdf")
    assert response.status_code == 200
    assert response.json()["failed_files"] == 1
    assert response.json()["successful_files"] == 1
    assert str(error) not in response.text
    for call in factory.get_ingestion_pipeline.return_value.ingest_pdf.call_args_list:
        assert not Path(call.args[0]).exists()


def test_folder_paths_and_duplicate_basenames(client, factory):
    response = upload_pdfs(client, "folder/a/paper.pdf", "folder/b/paper.pdf", "../../paper.pdf")
    body = response.json()
    assert body["successful_files"] == 3
    assert body["inserted_chunks"] == 21
    assert len({item["document_id"] for item in body["files"]}) == 3
    calls = factory.get_ingestion_pipeline.return_value.ingest_pdf.call_args_list
    for item, call in zip(body["files"], calls):
        target = Path(call.args[0])
        assert target.name == item["document_id"] + ".pdf"
        assert call.kwargs["metadata"]["source"] == item["relative_path"]
        assert call.kwargs["metadata"]["filename"] == "paper.pdf"
    assert len({Path(call.args[0]).parent for call in calls}) == 1


def test_upload_limits(client, factory, monkeypatch):
    import src.api.ingest_api as api
    monkeypatch.setattr(api, "MAX_FILES", 1)
    assert upload_pdfs(client, "a.pdf", "b.pdf").status_code == 422
    monkeypatch.setattr(api, "MAX_FILE_BYTES", 4)
    assert upload_pdfs(client, "large.pdf").json()["failed_files"] == 1
    factory.get_ingestion_pipeline.assert_not_called()


def test_upload_page(client):
    response = client.get("/ingestion")
    assert response.status_code == 200
    assert "webkitdirectory" in response.text
    assert "webkitRelativePath" in response.text


def test_chain_defaults_and_overrides(client, factory):
    response = client.post("/chain/generate", json={"question": " What method? "})
    assert response.status_code == 200
    assert response.json() == {"answer": "An answer"}
    generate = factory.get_rag_pipeline.return_value.generate
    generate.assert_called_with("What method?", rrf_k=12, rerank_k=3)
    response = client.post("/chain/generate", json={"question": "Why?", "rrf_k": 5, "rerank_k": 2})
    assert response.status_code == 200
    generate.assert_called_with("Why?", rrf_k=5, rerank_k=2)


@pytest.mark.parametrize("payload", [
    {}, {"question": " "}, {"question": "Q", "rrf_k": 0},
    {"question": "Q", "rrf_k": True}, {"question": "Q", "rrf_k": 2.5},
    {"question": "Q", "rrf_k": 2},  # Less than YAML rerank_k.
    {"question": "Q", "rerank_k": 13},
])
def test_invalid_chain_input(client, factory, payload):
    assert client.post("/chain/generate", json=payload).status_code == 422
    factory.get_rag_pipeline.assert_not_called()


def test_empty_corpus_can_be_ingested_then_queried(client, factory):
    factory.repository.count.return_value = 0
    assert client.post("/chain/generate", json={"question": "Q"}).status_code == 409
    factory.get_rag_pipeline.assert_not_called()
    assert upload_pdfs(client, "paper.pdf").json()["successful_files"] == 1
    factory.repository.count.return_value = 7
    assert client.post("/chain/generate", json={"question": "Q"}).status_code == 200


def test_rag_is_rebuilt_for_current_database_contents(client, factory):
    client.post("/chain/generate", json={"question": "Before?"})
    upload_pdfs(client, "paper.pdf")
    client.post("/chain/generate", json={"question": "After?"})
    assert factory.get_rag_pipeline.call_count == 2


def test_chain_backend_error_does_not_leak_details(client, factory):
    factory.get_rag_pipeline.return_value.generate.side_effect = RuntimeError("secret token")
    response = client.post("/chain/generate", json={"question": "Q"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Pipeline execution failed."}
