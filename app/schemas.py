"""
Shared Pydantic models (request/response schemas) used across the
API layer, services layer, and RAG layer.

Centralizing schemas here avoids duplicated model definitions and
keeps request/response contracts consistent across every endpoint.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    """Adaptive difficulty levels supported by the game."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# --------------------------------------------------------------------------
# /start-session
# --------------------------------------------------------------------------
class StartSessionRequest(BaseModel):
    lesson_id: str = Field(..., description="The lesson selected by the player in the frontend")
    student_name: Optional[str] = Field(None, description="Optional display name for the report")


class StartSessionResponse(BaseModel):
    session_id: str
    lesson_id: str
    current_difficulty: Difficulty
    message: str


# --------------------------------------------------------------------------
# /generate-question
# --------------------------------------------------------------------------
class GenerateQuestionRequest(BaseModel):
    session_id: str = Field(..., description="Active session identifier returned by /start-session")


class QuestionModel(BaseModel):
    """Schema of a single generated MCQ question."""

    type: str = Field(default="mcq")
    question: str
    choices: List[str] = Field(..., min_length=2)
    correct_answer: str
    difficulty: Difficulty


class GenerateQuestionResponse(BaseModel):
    session_id: str
    question: QuestionModel


# --------------------------------------------------------------------------
# /submit-answer
# --------------------------------------------------------------------------
class SubmitAnswerRequest(BaseModel):
    session_id: str
    question: str
    correct_answer: str
    player_answer: str
    time_taken: float = Field(..., ge=0, description="Time taken by the player, in seconds")


class EvaluationResult(BaseModel):
    is_correct: bool
    score: int
    difficulty: Difficulty
    time: float
    feedback: str


# --------------------------------------------------------------------------
# /finish-session
# --------------------------------------------------------------------------
class FinishSessionRequest(BaseModel):
    session_id: str


class QuestionAnalysis(BaseModel):
    question: str
    difficulty: Difficulty
    time: float
    correct: bool


class FinalReport(BaseModel):
    student_score: int
    correct_answers: int
    wrong_answers: int
    accuracy: float
    average_time: float
    highest_level: Difficulty
    strengths: List[str]
    weaknesses: List[str]
    common_mistakes: List[str]
    recommendations: List[str]
    question_analysis: List[QuestionAnalysis]


# --------------------------------------------------------------------------
# Internal session record (kept in memory, not exposed directly via API)
# --------------------------------------------------------------------------
class AnsweredQuestionRecord(BaseModel):
    question: str
    correct_answer: str
    player_answer: str
    difficulty: Difficulty
    time_seconds: float
    is_correct: bool
