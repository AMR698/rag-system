"""
Lesson-scoped retriever.

IMPORTANT: This module does NOT create embeddings and does NOT create
a new collection. It only CONNECTS to the existing, already-embedded
ChromaDB collection (created by a separate embedding pipeline) and
retrieves chunks that belong to a specific `lesson_id`.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """Raised when the lesson context cannot be retrieved."""


class LessonRetriever:
    """
    Loads the existing Chroma collection and exposes lesson-scoped
    retrieval methods.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings

        logger.info(
            "Connecting to existing Chroma store at %s",
            settings.CHROMA_DB_PATH,
        )

        try:
            # Connect to persistent ChromaDB
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_DB_PATH,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                ),
            )

            # Load existing collection
            self._collection = self._client.get_collection(
                name=settings.COLLECTION_NAME
            )

        except Exception as exc:
            logger.exception("Failed to initialize ChromaDB")
            raise RetrieverError(
                f"Could not connect to ChromaDB collection "
                f"'{settings.COLLECTION_NAME}': {exc}"
            ) from exc

        # Query embedding model
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )

        # LangChain wrapper
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=settings.COLLECTION_NAME,
            embedding_function=self._embeddings,
        )

        logger.info(
            "Connected successfully to collection '%s'",
            settings.COLLECTION_NAME,
        )

    def get_lesson_context(
        self,
        lesson_id: str,
        max_chunks: int = 50,
    ) -> str:
        """
        Retrieve all chunks belonging to a lesson.
        """

        try:
            result = self._collection.get(
                where={"lesson_id": lesson_id},
                limit=max_chunks,
                include=[
                    "documents",
                    "metadatas",
                    "ids",
                ],
            )

        except Exception as exc:
            logger.exception(
                "Chroma get() failed for lesson_id=%s",
                lesson_id,
            )
            raise RetrieverError(
                f"Failed to retrieve lesson '{lesson_id}': {exc}"
            ) from exc

        documents = result.get("documents") or []

        if not documents:
            logger.warning(
                "No chunks found for lesson_id=%s",
                lesson_id,
            )
            return ""

        merged = "\n\n".join(documents)

        logger.info(
            "Retrieved %d chunks for lesson_id=%s",
            len(documents),
            lesson_id,
        )

        return merged

    def semantic_search(
        self,
        lesson_id: str,
        query: str,
        top_k: int = 4,
    ) -> list[str]:
        """
        Semantic search restricted to a specific lesson.
        """

        try:
            docs = self._vectorstore.similarity_search(
                query=query,
                k=top_k,
                filter={"lesson_id": lesson_id},
            )

        except Exception as exc:
            logger.exception(
                "Semantic search failed for lesson_id=%s",
                lesson_id,
            )
            raise RetrieverError(
                f"Semantic search failed: {exc}"
            ) from exc

        return [doc.page_content for doc in docs]

    def lesson_exists(self, lesson_id: str) -> bool:
        """
        Check whether the lesson exists.
        """

        result = self._collection.get(
            where={"lesson_id": lesson_id},
            limit=1,
        )

        return bool(result.get("documents"))


@lru_cache
def get_retriever() -> LessonRetriever:
    """
    Return a singleton retriever instance.
    """
    return LessonRetriever()
