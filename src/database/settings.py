import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
CONFIGS_PATH = PROJECT_ROOT / "configs" / "vectordb_config.yml"

class Settings:
    def __init__(self, config_path: str = CONFIGS_PATH, env_path: str = ENV_PATH):
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)

        self._load_env()
        self.config = self._load_config()

    def _load_env(self) -> None:
        if not self.env_path.exists():
            raise FileNotFoundError(f"Not found env file: {self.env_path}")
        load_dotenv(self.env_path)

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Not found config file: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8" ) as file:
            config = yaml.safe_load(file)
        if config is None:
            raise ValueError("config.yml is empty")
        return config

    # VECTOR STORE
    @property
    def default_provider(self) -> str:
        return self.config["vector_store"]["default_provider"]

    # PGVECTOR
    @property
    def pgvector_db_url(self) -> str:
        value = os.getenv("PG_DATABASE_URL")
        if not value:
            raise ValueError("PG_DATABASE_URL is not set in .env")
        return value

    @property
    def pgvector_config(self) -> dict:
        return self.config["pgvector"]

    @property
    def pgvector_table_name(self) -> str:
        return self.pgvector_config["table_name"]

    # QDRANT
    @property
    def qdrant_url(self) -> str:
        value = os.getenv("QDRANT_DATABASE_URL")
        if not value:
            raise ValueError("QDRANT_DATABASE_URL is not set in .env")
        return value

    @property
    def qdrant_api_key(self):
        value = os.getenv("QDRANT_DATABASE_API_KEY")
        return value or None

    @property
    def qdrant_config(self) -> dict:
        return self.config["qdrant"]

    @property
    def qdrant_collection_name(self) -> str:
        return self.qdrant_config["collection_name"]

    # FAISS
    @property
    def faiss_config(self) -> dict:
        return self.config["faiss"]

    @property
    def faiss_index_path(self) -> str:
        return self.faiss_config[
            "index_path"
        ]

    @property
    def faiss_metadata_path(self) -> str:
        return self.faiss_config[
            "metadata_path"
        ]