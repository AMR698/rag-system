"""
FastAPI routes for the educational RAG game backend.

Endpoints:
    POST /start-session      -> begin a new game session for a lesson_id
    POST /generate-question  -> generate the next adaptive-difficulty question
    POST /submit-answer      -> evaluate a player's answer
    POST /finish-session     -> close the session and return the final report
    GET  /health             -> liveness/readiness probe
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.rag.generator import GenerationError, InsufficientContextError
from app.rag.retriever import RetrieverError
from app.schemas import (
    FinalReport,
    FinishSessionRequest,
    GenerateQuestionRequest,
    GenerateQuestionResponse,
    StartSessionRequest,
    StartSessionResponse,
    SubmitAnswerRequest,
    EvaluationResult,
)
from app.services.game_service import (
    GameService,
    LessonNotFoundError,
    SessionNotFoundError,
    get_game_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["system"])
def health_check() -> dict:
    """Simple liveness probe used by orchestrators / load balancers."""
    return {"status": "ok"}


@router.post("/start-session", response_model=StartSessionResponse, tags=["game"])
def start_session(
    payload: StartSessionRequest,
    game_service: GameService = Depends(get_game_service),
) -> StartSessionResponse:
    """
    Start a new game session scoped to a single lesson_id.
    The lesson's chunks are retrieved once and cached for the whole session.
    """
    try:
        session = game_service.start_session(
            lesson_id=payload.lesson_id,
            student_name=payload.student_name,
        )
    except LessonNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RetrieverError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return StartSessionResponse(
        session_id=session.session_id,
        lesson_id=session.lesson_id,
        current_difficulty=session.current_difficulty,
        message="Session started successfully.",
    )


@router.post("/generate-question", response_model=GenerateQuestionResponse, tags=["game"])
def generate_question(
    payload: GenerateQuestionRequest,
    game_service: GameService = Depends(get_game_service),
) -> GenerateQuestionResponse:
    """Generate the next question for the given session, at its current adaptive difficulty."""
    try:
        question = game_service.generate_question(payload.session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except InsufficientContextError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except GenerationError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return GenerateQuestionResponse(session_id=payload.session_id, question=question)


@router.post("/submit-answer", response_model=EvaluationResult, tags=["game"])
def submit_answer(
    payload: SubmitAnswerRequest,
    game_service: GameService = Depends(get_game_service),
) -> EvaluationResult:
    """Evaluate a player's answer and advance the adaptive difficulty state machine."""
    try:
        result = game_service.submit_answer(
            session_id=payload.session_id,
            question=payload.question,
            correct_answer=payload.correct_answer,
            player_answer=payload.player_answer,
            time_taken=payload.time_taken,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return result


@router.post("/finish-session", response_model=FinalReport, tags=["game"])
def finish_session(
    payload: FinishSessionRequest,
    game_service: GameService = Depends(get_game_service),
) -> FinalReport:
    """Close the session and return the full performance report as pure JSON."""
    try:
        report = game_service.finish_session(payload.session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return report
