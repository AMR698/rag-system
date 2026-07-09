"""
Application settings.

All configuration values are loaded from the `.env` file located at
`secrets/.env`. Nothing is ever hardcoded in the source code — this
keeps API keys and environment-specific paths out of version control
and makes the module portable across dev / staging / production.

Usage:
    from app.config.settings import get_settings
    settings = get_settings()
    print(settings.QWEN_API_KEY)
"""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Path to the .env file (kept under /secrets as required by the spec)
ENV_FILE_PATH = Path(__file__).resolve().parents[2] / "secrets" / ".env"


class Settings(BaseSettings):
    """
    Strongly-typed application settings.

    Every field maps 1:1 to a variable inside secrets/.env.
    """

    # --- LLM (Qwen) configuration ---
    QWEN_API_KEY: str = Field(default="", description="API key for the Qwen LLM provider")
    QWEN_BASE_URL: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="OpenAI-compatible base URL for the Qwen API",
    )
    QWEN_MODEL: str = Field(default="qwen-plus", description="Qwen model name to use for generation")

    # --- ChromaDB configuration ---
    CHROMA_DB_PATH: str = Field(default="./vector_db/chroma_store", description="Path to the persistent Chroma store")
    COLLECTION_NAME: str = Field(default="education_kb", description="Name of the Chroma collection holding lesson embeddings")

    # --- Embeddings ---
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformers model used to embed queries against the existing collection",
    )

    # --- App-level ---
    APP_NAME: str = "Educational RAG Game Backend"
    LOG_LEVEL: str = Field(default="INFO", description="Root logging level")

    # --- Adaptive difficulty tuning ---
    STREAK_TO_LEVEL_UP: int = Field(default=5, description="Consecutive correct answers required to level up")
    MISTAKES_TO_LEVEL_DOWN: int = Field(default=3, description="Consecutive wrong answers required to level down")

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    lru_cache ensures the .env file is parsed only once per process,
    and the same Settings object is reused (dependency-injection friendly).
    """
    return Settings()


def configure_logging() -> None:
    """Configure root logging based on the LOG_LEVEL setting."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
