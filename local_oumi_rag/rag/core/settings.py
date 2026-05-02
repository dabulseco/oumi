from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings with env-var overrides.

    Prefix: ``LOCAL_OUMI_RAG_``
    Example: ``LOCAL_OUMI_RAG_MILVUS_URI=http://localhost:19530``
    """

    model_config = SettingsConfigDict(env_prefix="LOCAL_OUMI_RAG_", extra="ignore")

    repo_root: Path = Field(default=Path("."))
    cache_dir: Path = Field(default=Path("local_oumi_rag/.cache"))
    max_file_size_bytes: int = Field(default=2_000_000)

    milvus_uri: str = Field(default="http://localhost:19530")
    milvus_collection: str = Field(default="oumi_repo_chunks")

    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    top_k_dense: int = Field(default=24)
    top_k_sparse: int = Field(default=24)
    top_k_final: int = Field(default=8)
    dense_weight: float = Field(default=0.65)
    sparse_weight: float = Field(default=0.35)

    default_ollama_model: str = Field(default="llama3.1:8b")


def get_settings() -> Settings:
    return Settings()
