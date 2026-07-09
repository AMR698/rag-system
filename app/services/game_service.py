"""
Game service.

Orchestrates the full game loop on top of the RAG layer:
    - session lifecycle (start / finish)
    - adaptive difficulty progression
    - in-memory tracking of every answered question
    - delegating to retriever / generator / evaluator / report

No SQL database is used, as required by the spec — everything lives
in an in-memory dict keyed by session_id. This is appropriate for a
single-process game backend; swapping in Redis/SQL later would only
require changing this class's internal storage, not its public API.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from app.config.settings import get_settings
from app.rag.evaluator import AnswerEvaluator
from app.rag.generator import GenerationError, InsufficientContextError, QuestionGenerator
from app.rag.report import ReportGenerator
from app.rag.retriever import LessonRetriever, RetrieverError, get_retriever
from app.schemas import (
    AnsweredQuestionRecord,
    Difficulty,
    EvaluationResult,
    FinalReport,
    QuestionModel,
)

logger = logging.getLogger(__name__)

_DIFFICULTY_SEQUENCE = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]


class SessionNotFoundError(Exception):
    """Raised when an unknown session_id is used."""


class LessonNotFoundError(Exception):
    """Raised when the requested lesson_id has no chunks in the vector DB."""


@dataclass
class GameSession:
    """In-memory representation of one player's game session."""

    session_id: str
    lesson_id: str
    student_name: Optional[str] = None

    current_difficulty: Difficulty = Difficulty.EASY
    correct_streak: int = 0
    wrong_streak: int = 0

    lesson_context: str = ""
    asked_questions: List[str] = field(default_factory=list)
    # Keep the last generated question object so /submit-answer can validate against it.
    last_question: Optional[QuestionModel] = None

    answered_records: List[AnsweredQuestionRecord] = field(default_factory=list)


class GameService:
    """
    High-level orchestrator used directly by the FastAPI route handlers.
    One instance is shared for the whole process (see get_game_service()).
    """

    def __init__(self, retriever: LessonRetriever) -> None:
        self._retriever = retriever
        self._generator = QuestionGenerator()
        self._evaluator = AnswerEvaluator()
        self._report_generator = ReportGenerator()
        self._sessions: Dict[str, GameSession] = {}
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def start_session(self, lesson_id: str, student_name: Optional[str] = None) -> GameSession:
        if not self._retriever.lesson_exists(lesson_id):
            raise LessonNotFoundError(f"No content found for lesson_id='{lesson_id}'.")

        lesson_context = self._retriever.get_lesson_context(lesson_id)

        session = GameSession(
            session_id=str(uuid.uuid4()),
            lesson_id=lesson_id,
            student_name=student_name,
            lesson_context=lesson_context,
        )
        self._sessions[session.session_id] = session
        logger.info("Started session %s for lesson_id=%s", session.session_id, lesson_id)
        return session

    def _get_session(self, session_id: str) -> GameSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session '{session_id}' not found or already finished.")
        return session

    def finish_session(self, session_id: str) -> FinalReport:
        session = self._get_session(session_id)
        report = self._report_generator.generate(session.answered_records)
        # Free the session's memory once the report has been produced.
        del self._sessions[session_id]
        logger.info("Finished session %s", session_id)
        return report

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------
    def generate_question(self, session_id: str) -> QuestionModel:
        session = self._get_session(session_id)

        try:
            question = self._generator.generate(
                lesson_context=session.lesson_context,
                difficulty=session.current_difficulty,
                previous_questions=session.asked_questions,
            )
        except InsufficientContextError:
            logger.warning("Insufficient context for lesson_id=%s", session.lesson_id)
            raise
        except GenerationError:
            logger.exception("Question generation failed for session %s", session_id)
            raise

        session.last_question = question
        session.asked_questions.append(question.question)
        return question

    # ------------------------------------------------------------------
    # Answer submission + adaptive difficulty
    # ------------------------------------------------------------------
    def submit_answer(
        self,
        session_id: str,
        question: str,
        correct_answer: str,
        player_answer: str,
        time_taken: float,
    ) -> EvaluationResult:
        session = self._get_session(session_id)

        result = self._evaluator.evaluate(
            question=question,
            correct_answer=correct_answer,
            player_answer=player_answer,
            difficulty=session.current_difficulty,
            time_taken=time_taken,
        )

        # Track the full record (spec: no SQL, keep in memory).
        session.answered_records.append(
            AnsweredQuestionRecord(
                question=question,
                correct_answer=correct_answer,
                player_answer=player_answer,
                difficulty=session.current_difficulty,
                time_seconds=time_taken,
                is_correct=result.is_correct,
            )
        )

        self._update_difficulty(session, result.is_correct)
        return result

    def _update_difficulty(self, session: GameSession, was_correct: bool) -> None:
        """
        Adaptive difficulty state machine:
            - N consecutive correct answers -> level up (capped at Hard).
            - M consecutive wrong answers -> level down (floored at Easy).
        """
        settings = self._settings
        current_index = _DIFFICULTY_SEQUENCE.index(session.current_difficulty)

        if was_correct:
            session.correct_streak += 1
            session.wrong_streak = 0
            if session.correct_streak >= settings.STREAK_TO_LEVEL_UP and current_index < len(_DIFFICULTY_SEQUENCE) - 1:
                current_index += 1
                session.current_difficulty = _DIFFICULTY_SEQUENCE[current_index]
                session.correct_streak = 0
                logger.info(
                    "Session %s leveled UP to %s", session.session_id, session.current_difficulty
                )
        else:
            session.wrong_streak += 1
            session.correct_streak = 0
            if session.wrong_streak >= settings.MISTAKES_TO_LEVEL_DOWN and current_index > 0:
                current_index -= 1
                session.current_difficulty = _DIFFICULTY_SEQUENCE[current_index]
                session.wrong_streak = 0
                logger.info(
                    "Session %s leveled DOWN to %s", session.session_id, session.current_difficulty
                )


@lru_cache
def get_game_service() -> GameService:
    """Return a cached, process-wide GameService instance (dependency-injection friendly)."""
    return GameService(retriever=get_retriever())
