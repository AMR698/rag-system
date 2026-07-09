"""
Lesson-scoped retriever.

IMPORTANT: This module does NOT create embeddings and does NOT create
a new collection. It only CONNECTS to the existing, already-embedded
ChromaDB collection (created by a separate embedding pipeline) and
retrieves chunks that belong to a specific `lesson_id`.

Two retrieval strategies are exposed:
    1. get_lesson_context(lesson_id)  -> returns ALL chunks for that
       lesson, concatenated. Used to build the full context sent to
       the question generator.
    2. semantic_search(lesson_id, query, top_k) -> returns the most
       relevant chunks for a specific query, still scoped to the
       lesson via a metadata filter. Useful if the lesson is very
       large and only a slice of it is relevant to a sub-topic.

LangChain's Chroma vectorstore wrapper is used on top of the native
chromadb PersistentClient so that the same embedding function
(SentenceTransformers) used at ingestion time is reused consistently
for any query-time embedding.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """Raised when the lesson context cannot be retrieved."""


class LessonRetriever:
    """
    Loads the existing Chroma collection and exposes lesson-scoped
    retrieval methods. Embeddings are NEVER recreated here.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings

        logger.info("Connecting to existing Chroma store at %s", settings.CHROMA_DB_PATH)
        self._client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

        # Load the already-existing collection (do NOT create a new one).
        self._collection = self._client.get_collection(name=settings.COLLECTION_NAME)

        # Embedding function reused only for query-time embedding (LangChain wrapper).
        self._embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

        self._vectorstore = Chroma(
            client=self._client,
            collection_name=settings.COLLECTION_NAME,
            embedding_function=self._embeddings,
        )

    def get_lesson_context(self, lesson_id: str, max_chunks: int = 50) -> str:
        """
        Retrieve ALL chunks whose metadata["lesson_id"] == lesson_id
        and merge them into a single context string.

        Args:
            lesson_id: The lesson selected by the player.
            max_chunks: Safety cap on how many chunks to merge, to
                avoid overflowing the LLM context window.

        Returns:
            A single string with all lesson chunks concatenated, or an
            empty string if no chunks were found for that lesson.
        """
        try:
            result = self._collection.get(
                where={"lesson_id": lesson_id},
                limit=max_chunks,
                include=["documents", "metadatas"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Chroma get() failed for lesson_id=%s", lesson_id)
            raise RetrieverError(f"Failed to retrieve lesson '{lesson_id}': {exc}") from exc

        documents = result.get("documents") or []
        if not documents:
            logger.warning("No chunks found for lesson_id=%s", lesson_id)
            return ""

        merged = "\n\n".join(documents)
        logger.info("Retrieved %d chunks for lesson_id=%s", len(documents), lesson_id)
        return merged

    def semantic_search(self, lesson_id: str, query: str, top_k: int = 4) -> list[str]:
        """
        Retrieve the top_k most relevant chunks for `query`, restricted
        to the given lesson_id via a metadata filter.

        Useful for large lessons where only a portion of the content
        is relevant to a specific sub-topic or follow-up question.
        """
        try:
            docs = self._vectorstore.similarity_search(
                query=query,
                k=top_k,
                filter={"lesson_id": lesson_id},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Semantic search failed for lesson_id=%s", lesson_id)
            raise RetrieverError(f"Semantic search failed: {exc}") from exc

        return [doc.page_content for doc in docs]

    def lesson_exists(self, lesson_id: str) -> bool:
        """Quick check used by the API layer to validate a lesson_id before starting a session."""
        result = self._collection.get(where={"lesson_id": lesson_id}, limit=1)
        return bool(result.get("documents"))


@lru_cache
def get_retriever() -> LessonRetriever:
    """Return a cached, process-wide LessonRetriever instance (dependency-injection friendly)."""
    return LessonRetriever()
