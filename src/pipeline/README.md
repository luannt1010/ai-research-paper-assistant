# PipelineFactory

```python
from src.pipeline import PipelineFactory

# The default path is resolved from the project, regardless of working directory.
factory = PipelineFactory()  # Or PipelineFactory("path/to/app_configs.yml")
try:
    ingestor = factory.get_ingestion_pipeline()
    ingestor.ingest_pdf("papers/paper.pdf")

    rag = factory.get_rag_pipeline()
    answer = rag.generate("What method does this paper propose?")  # str
    answer = rag.generate("What are its limitations?", rrf_k=20, rerank_k=5)
finally:
    factory.close()
```

For an API, create the factory and pipelines during application startup and call
`factory.close()` at shutdown. Components are lazy and reused within a factory.
Construction of a RAG pipeline fits BM25 using the current database contents;
create it again after ingestion to include newly inserted documents. The existing
BM25 implementation requires a non-empty corpus to answer questions.

`factory.embedder`, `.llm`, `.reranker`, `.chunker`, `.loader`, and `.repository`
are also available individually. You can inject a repository with
`PipelineFactory(repository=repository)`; the caller is responsible for closing it.
Without injection, the factory uses `configs/vectordb_config.yml` and `.env` through
the existing database `Settings` class.

The sample app config uses existing Ollama wrappers. Embedding dimensions must
match the database vector column. Provider-specific Ollama arguments belong in
`options`. The factory creates clients; it does not download Ollama models.
Unknown providers and unsupported top-level component keys raise `ValueError`.

Supported chunking configurations:

```yaml
# Character lengths, using the existing chunker's separators.
chunking:
  strategy: recursive
  length_unit: character
  chunk_size: 1200
  chunk_overlap: 120

# Alternatively:
# chunking:
#   strategy: semantic
#   threshold: 95
#
# chunking:
#   strategy: hierarchical
#   length_unit: character
#   parent_chunk_size: 1200
#   parent_overlap: 120
#   child_chunk_size: 300
#   child_overlap: 30
```

Semantic chunking receives the underlying LangChain embedding client. Token-based
chunking and the unfinished Markdown chunker are not supported by this factory.
`retrieval.rrf_k` controls the number of retrieved results before reranking.

Pipelines are synchronous. Run their work in synchronous API handlers or a worker
thread rather than blocking an async event loop. The factory does not add
concurrency guarantees to the existing shared database connection or reranker;
serialize access to a shared pipeline or use isolated instances per worker.
