from functools import cached_property
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "app_configs.yml"


class PipelineFactory:
    """Reuse components within one factory; build a fresh retriever per RAG pipeline.

    Create during API startup. A supplied repository remains owned by the caller.
    After ingestion, rebuild the RAG pipeline to refresh its in-memory BM25 index.
    """

    def __init__(self, config_path=DEFAULT_CONFIG_PATH, *, repository=None):
        self.config_path = Path(config_path)
        with self.config_path.open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        self._provided_repository = repository
        self._validate_config()

    def _validate_config(self):
        if not isinstance(self.config, dict):
            raise ValueError("App config must be a YAML mapping.")
        allowed = {"embedding", "llm", "reranker", "chunking", "retrieval"}
        self._check_keys(self.config, allowed, "config")
        for name in allowed - {"retrieval"}:
            if not isinstance(self.config.get(name), dict):
                raise ValueError(f"{name} must be a YAML mapping.")

        for name, supported in (
            ("embedding", "ollama"),
            ("llm", "ollama"),
            ("reranker", "sentence_transformers"),
        ):
            section = self.config[name]
            if section.get("provider") != supported:
                raise ValueError(f"{name}.provider must be '{supported}'.")
            if not isinstance(section.get("model_name"), str) or not section["model_name"].strip():
                raise ValueError(f"{name}.model_name must be a non-empty string.")
            if name == "reranker":
                self._check_keys(section, {
                    "provider", "model_name", "device", "batch_size", "max_length",
                }, name)
                for key in ("batch_size", "max_length"):
                    if key in section:
                        self._positive_int(section[key], f"{name}.{key}")
            else:
                keys = {"provider", "model_name", "options"}
                if name == "llm":
                    keys.add("temperature")
                self._check_keys(section, keys, name)
                options = section.get("options", {})
                if not isinstance(options, dict) or not all(isinstance(k, str) for k in options):
                    raise ValueError(f"{name}.options must be a mapping with string keys.")
                if {"model", "model_name", "temperature"} & options.keys():
                    raise ValueError(f"Set model_name/temperature directly under {name}, not options.")

        chunking = self.config["chunking"]
        strategy = chunking.get("strategy")
        if strategy == "recursive":
            self._check_keys(chunking, {"strategy", "length_unit", "chunk_size", "chunk_overlap"}, "chunking")
            self._validate_chunk_pair(chunking, "chunk_size", "chunk_overlap")
        elif strategy == "hierarchical":
            self._check_keys(chunking, {
                "strategy", "length_unit", "parent_chunk_size", "parent_overlap",
                "child_chunk_size", "child_overlap",
            }, "chunking")
            self._validate_chunk_pair(chunking, "parent_chunk_size", "parent_overlap")
            self._validate_chunk_pair(chunking, "child_chunk_size", "child_overlap")
        elif strategy == "semantic":
            self._check_keys(chunking, {"strategy", "threshold"}, "chunking")
            threshold = chunking.get("threshold")
            if type(threshold) not in (int, float) or not 0 < threshold < 100:
                raise ValueError("chunking.threshold must be between 0 and 100 (exclusive).")
        else:
            raise ValueError("chunking.strategy must be recursive, hierarchical, or semantic.")
        if strategy != "semantic" and chunking.get("length_unit", "character") != "character":
            raise ValueError("Existing recursive/hierarchical chunkers support character lengths only.")

        retrieval = self.config.get("retrieval", {})
        if not isinstance(retrieval, dict):
            raise ValueError("retrieval must be a YAML mapping.")
        self._check_keys(retrieval, {"rrf_k", "rerank_k"}, "retrieval")
        rrf_k = retrieval.get("rrf_k", 10)
        rerank_k = retrieval.get("rerank_k", 5)
        self._positive_int(rrf_k, "retrieval.rrf_k")
        self._positive_int(rerank_k, "retrieval.rerank_k")
        if rerank_k > rrf_k:
            raise ValueError("retrieval.rerank_k must not exceed retrieval.rrf_k.")

    @staticmethod
    def _check_keys(section, allowed, name):
        unknown = section.keys() - allowed
        if unknown:
            raise ValueError(f"Unsupported keys in {name}: {', '.join(sorted(map(str, unknown)))}")

    @staticmethod
    def _positive_int(value, name):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @classmethod
    def _validate_chunk_pair(cls, section, size_key, overlap_key):
        size = section.get(size_key)
        overlap = section.get(overlap_key)
        cls._positive_int(size, f"chunking.{size_key}")
        if type(overlap) is not int or not 0 <= overlap < size:
            raise ValueError(f"chunking.{overlap_key} must satisfy 0 <= overlap < {size_key}.")

    @cached_property
    def embedder(self):
        from src.embedder import OllamaEmbedder

        config = self.config["embedding"]
        return OllamaEmbedder(config["model_name"], **config.get("options", {}))

    @cached_property
    def llm(self):
        from src.generation import OllamaLLM

        config = self.config["llm"]
        return OllamaLLM(
            model_name=config["model_name"],
            temperature=config.get("temperature", 0.0),
            **config.get("options", {}),
        )

    @cached_property
    def reranker(self):
        from src.reranker import CrossEncoderReRanker

        return CrossEncoderReRanker(**{
            key: value for key, value in self.config["reranker"].items()
            if key != "provider"
        })

    @cached_property
    def chunker(self):
        config = self.config["chunking"]
        params = {k: v for k, v in config.items() if k not in {"strategy", "length_unit"}}
        if config["strategy"] == "recursive":
            from src.chunker import RecursiveChunk
            return RecursiveChunk(**params)
        if config["strategy"] == "hierarchical":
            from src.chunker.hierarchical_chunk import HierarchicalChunker
            return HierarchicalChunker(**params)
        from src.chunker import SemanticChunk
        return SemanticChunk(embedding_model=self.embedder.embedder, **params)

    @cached_property
    def loader(self):
        from src.ingestion.parser import SimpleLoader
        return SimpleLoader()

    @cached_property
    def repository(self):
        if self._provided_repository is not None:
            return self._provided_repository
        from src.database.settings import Settings
        from src.database import PGVectorStore, QdrantVectorStore

        provider = Settings().default_provider
        stores = {"pgvector": PGVectorStore, "qdrant": QdrantVectorStore}
        if provider not in stores:
            raise ValueError(f"Unsupported vector store provider: {provider}")
        return stores[provider]()

    def get_ingestion_pipeline(self):
        from .ingestion_pipeline import IngestionPipeline
        return IngestionPipeline(self.loader, self.chunker, self.embedder, self.repository)

    def get_rag_pipeline(self):
        from .rag_pipeline import RAGChain
        return RAGChain(
            llm=self.llm,
            repository=self.repository,
            embedder=self.embedder,
            reranker=self.reranker,
            **self.config.get("retrieval", {}),
        )

    def close(self):
        """Close only a repository created by this factory, e.g. on API shutdown."""
        if self._provided_repository is None and "repository" in self.__dict__:
            self.__dict__.pop("repository").close()
